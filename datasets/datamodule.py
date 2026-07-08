#!/usr/bin/env python3
"""数据加载模块"""

import hashlib
import json
import warnings
import logging
import os
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Callable

if TYPE_CHECKING:
    from text_encoder import TextEncoderConfig

import torch
import lightning as L
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader

from .asm_encoding import load_asm_encodings as _load_asm_encodings
from .data_types import DualGraphData
from .graph_builders import (
    build_asm_graph as _build_asm_graph,
    build_rtl_structure as _build_rtl_structure,
    empty_asm_graph as _empty_asm_graph,
)
from .manifest import (
    DatasetDirInput,
    normalize_dataset_dirs as _normalize_dataset_dirs,
    read_manifest_samples_from_dirs as _read_manifest_samples_from_dirs,
)
from .samplers import RtlSizeBucketSampler
from .targets import extract_targets_from_sample as _extract_targets_from_sample

logger = logging.getLogger(__name__)


def _build_and_save_labels(
    args: tuple[dict, str, str, str, int],
) -> bool:
    """阶段 2 worker: 仅提取标签并保存轻量 data_{idx}.pt

    RTL 图结构和 ASM 图已去重保存，此处仅保存 edge_labels + y + 引用文件名。
    通过 _valid_edge_mask 保证标签与图结构的边过滤一致。
    """
    sample, rtl_filename, asm_filename, processed_dir, idx = args

    try:
        rtl_json_path = sample["_rtl_path"]
        with open(rtl_json_path, encoding="utf-8") as f:
            rtl = json.load(f)

        valid_edge_mask = _build_rtl_structure(rtl)["_valid_edge_mask"]
        edge_labels, y = _extract_targets_from_sample(sample, valid_edge_mask)

        data = DualGraphData(
            edge_labels=edge_labels,
            y=y,
            _rtl_file=rtl_filename,
            _asm_file=asm_filename,
        )

        pt_path = Path(processed_dir) / f"data_{idx}.pt"
        torch.save(data, pt_path)
        return True

    except Exception as e:
        logger.error("构建/保存标签失败 [%s]: %s", sample.get("_rtl_path"), e)
        return False


class DualGraphDataset(Dataset):
    """
    双图数据集 - 继承自 PyG Dataset

    从 coverage-report-extractor 生成的数据集目录加载数据。
    目录须包含 manifest.json 和 samples.jsonlines。

    Args:
        root: 数据集根目录，包含 processed/ 子目录（PyG 约定）
        dataset_dir: coverage-report-extractor 输出目录（含 manifest.json）
        text_encoder_config: 文本编码器配置，None 时回退到零向量
        transform: 每次获取数据时应用的变换
        pre_transform: 处理前应用的变换（保存到磁盘）
        pre_filter: 处理前的过滤函数
    """

    def __init__(
        self,
        root: Optional[str] = "dataset_root",
        dataset_dir: Optional[DatasetDirInput] = None,
        text_encoder_config: Optional["TextEncoderConfig"] = None,
        num_workers: int = 0,
        asm_chunk_files: int = 64,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.root = root
        self._dataset_dirs = _normalize_dataset_dirs(dataset_dir)
        self._num_samples: Optional[int] = None
        self._idx_to_file: list[str] = []
        self._text_encoder_config = text_encoder_config
        self._num_workers = num_workers or min(os.cpu_count() or 1, 32)
        self._asm_chunk_files = max(1, int(asm_chunk_files))
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> List[str]:
        if not self._dataset_dirs:
            return []
        return [
            str(dataset_dir / "manifest.json") for dataset_dir in self._dataset_dirs
        ]

    def _scan_processed_files(self) -> list[str]:
        processed_path = Path(self.processed_dir)
        if processed_path.exists():
            files = sorted(
                f.name
                for f in processed_path.iterdir()
                if f.name.startswith("data_") and f.name.endswith(".pt")
            )
            if files:
                self._num_samples = len(files)
                self._idx_to_file = files
                return files
        return []

    @property
    def processed_file_names(self) -> List[str]:
        if self._idx_to_file:
            return self._idx_to_file
        return self._scan_processed_files()

    def download(self):
        pass

    def process(self):
        """两阶段流水线：批量 GPU 编码 → 多进程张量构建"""
        if not self._dataset_dirs:
            logger.warning("未指定 dataset_dir，跳过处理")
            self._num_samples = 0
            return

        for dataset_dir in self._dataset_dirs:
            manifest_file = dataset_dir / "manifest.json"
            if not manifest_file.exists():
                logger.error("manifest.json 不存在: %s", manifest_file)
                self._num_samples = 0
                return

        try:
            samples = _read_manifest_samples_from_dirs(self._dataset_dirs)
        except Exception as e:
            logger.error("读取 manifest/samples 失败: %s", e)
            self._num_samples = 0
            return
        if not samples:
            logger.warning("samples.jsonlines 为空")
            self._num_samples = 0
            return

        logger.info(
            "读取 manifest 完成: %d 个数据集, %d 个样本",
            len(self._dataset_dirs),
            len(samples),
        )

        # ── 阶段 1: 批量 GPU 编码（CodeBERT）+ 磁盘缓存 ──
        unique_asm_paths = sorted(
            {
                sample["_asm_path"]
                for sample in samples
                if sample.get("_asm_path") is not None
            }
        )
        asm_encodings = _load_asm_encodings(
            unique_asm_paths,
            cache_root=self.root,
            text_encoder_config=self._text_encoder_config,
            asm_chunk_files=self._asm_chunk_files,
        )

        # ── 阶段 1.5: 构建并保存去重 RTL 图 + ASM 图 ──
        processed_dir = self.processed_dir

        # --- RTL 图去重 ---
        rtl_path_to_file: dict[str, str] = {}
        for sample in samples:
            rtl_path = sample["_rtl_path"]
            if rtl_path in rtl_path_to_file:
                continue
            module_name = sample.get("module_name") or Path(rtl_path).stem
            graph_key = hashlib.sha256(rtl_path.encode("utf-8")).hexdigest()[:16]
            rtl_filename = f"rtl_{module_name}_{graph_key}.pt"
            rtl_file_path = Path(processed_dir) / rtl_filename
            if not rtl_file_path.exists():
                with open(rtl_path, encoding="utf-8") as f:
                    rtl_json = json.load(f)
                rtl_data = _build_rtl_structure(rtl_json)
                rtl_data.pop("_valid_edge_mask", None)
                torch.save(rtl_data, rtl_file_path)
            rtl_path_to_file[rtl_path] = rtl_filename

        logger.info("阶段 1.5a: 保存 %d 个去重 RTL 图结构", len(rtl_path_to_file))

        # --- ASM 图去重 ---
        asm_path_to_file: dict[Optional[str], str] = {}
        empty_asm_filename = "asm_empty.pt"
        empty_asm_path = Path(processed_dir) / empty_asm_filename
        if not empty_asm_path.exists():
            torch.save(_empty_asm_graph(), empty_asm_path)
        asm_path_to_file[None] = empty_asm_filename

        for asm_path in unique_asm_paths:
            asm_hash = hashlib.sha256(asm_path.encode("utf-8")).hexdigest()[:16]
            asm_filename = f"asm_{asm_hash}.pt"
            asm_file_path = Path(processed_dir) / asm_filename
            if not asm_file_path.exists():
                asm_encoding = asm_encodings.get(asm_path)
                asm_data = _build_asm_graph(asm_path, asm_encoding)
                torch.save(asm_data, asm_file_path)
            asm_path_to_file[asm_path] = asm_filename

        logger.info("阶段 1.5b: 保存 %d 个去重 ASM 图", len(asm_path_to_file))

        del asm_encodings

        # ── 阶段 2: 多进程标签提取 + 保存（轻量） ──
        save_args: list[tuple[dict, str, str, str, int]] = []

        for i, sample in enumerate(samples):
            rtl_path = sample["_rtl_path"]
            asm_path = sample.get("_asm_path")
            rtl_filename = rtl_path_to_file.get(rtl_path)
            asm_filename = asm_path_to_file.get(asm_path)
            if rtl_filename is None or asm_filename is None:
                logger.error("图文件缺失，跳过: rtl=%s asm=%s", rtl_path, asm_path)
                continue
            save_args.append((sample, rtl_filename, asm_filename, processed_dir, i))

        logger.info(
            "阶段 2/2: 多进程标签提取 + 保存 (%d 个样本, %d workers)",
            len(save_args),
            self._num_workers,
        )
        with Pool(self._num_workers) as pool:
            results = pool.map(_build_and_save_labels, save_args)

        success_count = sum(1 for r in results if r)

        # 紧缩索引，消除失败样本导致的缺口
        actual_idx = 0
        for i, r in enumerate(results):
            if r:
                orig_idx = save_args[i][4]
                if orig_idx != actual_idx:
                    src = Path(processed_dir) / f"data_{orig_idx}.pt"
                    dst = Path(processed_dir) / f"data_{actual_idx}.pt"
                    src.rename(dst)
                actual_idx += 1

        self._num_samples = success_count
        self._idx_to_file = [f"data_{i}.pt" for i in range(success_count)]
        logger.info("数据处理完成: %d/%d 个样本成功", success_count, len(save_args))

    def len(self) -> int:
        if self._num_samples is not None:
            return self._num_samples
        files = self._scan_processed_files()
        return len(files)

    def get(self, idx: int) -> DualGraphData:
        """从磁盘加载指定索引的数据，透明合并去重的 RTL/ASM 图"""
        if self._idx_to_file:
            filename = self._idx_to_file[idx]
        else:
            filename = f"data_{idx}.pt"
        data = torch.load(Path(self.processed_dir) / filename, weights_only=False)

        rtl_file = getattr(data, "_rtl_file", None)
        if rtl_file is not None:
            rtl = self._load_cached(rtl_file)
            for k, v in rtl.items():
                setattr(data, k, v)
            del data._rtl_file

        asm_file = getattr(data, "_asm_file", None)
        if asm_file is not None:
            asm = self._load_cached(asm_file)
            for k, v in asm.items():
                setattr(data, k, v)
            del data._asm_file

        return data

    def _load_cached(self, filename: str) -> dict:
        if not hasattr(self, "_file_cache"):
            self._file_cache: dict[str, dict] = {}
        if filename not in self._file_cache:
            self._file_cache[filename] = torch.load(
                Path(self.processed_dir) / filename, weights_only=False
            )
        return self._file_cache[filename]


class DualGraphDataModule(L.LightningDataModule):
    """双图数据加载模块"""

    def __init__(
        self,
        root: str,
        dataset_dir: Optional[DatasetDirInput] = None,
        text_encoder_config: Optional["TextEncoderConfig"] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        asm_chunk_files: int = 64,
        transform: Optional[Callable] = None,
        use_bucketing: bool = False,
        token_budget: int = 8192,
    ):
        """
        Args:
            root: 数据集根目录（存放 processed/ .pt 文件）
            dataset_dir: coverage-report-extractor 输出目录（含 manifest.json）
            text_encoder_config: 文本编码器配置，None 时回退到零向量
            batch_size: 固定 batch 大小（use_bucketing=False 时生效）
            num_workers: 数据加载线程数
            asm_chunk_files: CodeBERT 编码时每批读取的 ASM 文件数
            transform: 数据变换
            use_bucketing: 是否启用按 RTL 图大小分桶的动态 batch（默认 False）
            token_budget: bucketing 模式下每 batch 最大节点总数
        """
        super().__init__()
        self.root = root
        self.dataset_dir = dataset_dir
        self.text_encoder_config = text_encoder_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.asm_chunk_files = max(1, int(asm_chunk_files))
        self.transform = transform
        self.use_bucketing = use_bucketing
        self.token_budget = token_budget

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _make_dataset(self, split: str) -> DualGraphDataset:
        return DualGraphDataset(
            root=str(Path(self.root) / split),
            dataset_dir=self.dataset_dir,
            text_encoder_config=self.text_encoder_config,
            num_workers=self.num_workers,
            asm_chunk_files=self.asm_chunk_files,
            transform=self.transform,
        )

    def prepare_data(self):
        """在 DDP 初始化之前完成全部数据处理（仅 rank 0）"""
        self._make_dataset("train")

    def setup(self, stage: Optional[str] = None):
        """初始化数据集

        val 集缺失时自动从 train 集按 9:1 划分。
        """
        root_path = Path(self.root)
        if stage == "fit" or stage is None:
            self.train_dataset = self._make_dataset("train")
            if (root_path / "val" / "processed").exists():
                self.val_dataset = self._make_dataset("val")
            else:
                from torch.utils.data import random_split

                full_len = len(self.train_dataset)
                val_len = max(1, int(full_len * 0.1))
                train_len = full_len - val_len
                split_gen = torch.Generator().manual_seed(42)
                self.train_dataset, self.val_dataset = random_split(
                    self.train_dataset,
                    [train_len, val_len],
                    generator=split_gen,
                )
                logger.info(
                    "val 集缺失，自动划分: train=%d, val=%d", train_len, val_len
                )

        if stage == "test" or stage is None:
            if (root_path / "test" / "processed").exists():
                self.test_dataset = self._make_dataset("test")

    def _create_loader(self, dataset, shuffle: bool) -> PyGDataLoader:
        n_samples = len(dataset)
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        per_gpu = n_samples // max(world_size, 1)

        if per_gpu < self.batch_size * 2:
            effective_workers = 0
            warnings.filterwarnings(
                "ignore",
                message=".*does not have many workers.*",
                module=r"lightning\.pytorch\.trainer\.connectors\.data_connector",
            )
        else:
            effective_workers = min(self.num_workers, per_gpu)
        persistent = effective_workers > 0 and n_samples > effective_workers

        if self.use_bucketing:
            batch_sampler = RtlSizeBucketSampler(
                dataset,
                token_budget=self.token_budget,
                shuffle=shuffle,
            )
            return PyGDataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=effective_workers,
                follow_batch=["asm_node_type"],
                pin_memory=True,
                persistent_workers=persistent,
            )

        return PyGDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=effective_workers,
            follow_batch=["asm_node_type"],
            pin_memory=True,
            persistent_workers=persistent,
        )

    def train_dataloader(self) -> PyGDataLoader:
        return self._create_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> Optional[PyGDataLoader]:
        if self.val_dataset is None:
            return None
        return self._create_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> Optional[PyGDataLoader]:
        if self.test_dataset is None:
            return None
        return self._create_loader(self.test_dataset, shuffle=False)
