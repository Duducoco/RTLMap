#!/usr/bin/env python3
"""数据加载模块"""

import hashlib
import json
import warnings

import orjson
from tqdm import tqdm
import logging
import os
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Callable, Union

if TYPE_CHECKING:
    from text_encoder import TextEncoderConfig

import torch
import lightning as L
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader

from cdfg_rtl.data_types import EdgeType, get_cell_type_index, CELL_TYPE_TO_NODE_TYPE
from cdfg_asm.data_types import AsmNodeType, AsmEdgeType
from .data_types import DualGraphData

logger = logging.getLogger(__name__)

# 枚举名 → 0-indexed 映射表
_EDGE_TYPE_MAP = {e.name: e.value - 1 for e in EdgeType}
_ASM_NODE_TYPE_MAP = {e.name: e.value - 1 for e in AsmNodeType}
_ASM_EDGE_TYPE_MAP = {e.name: e.value - 1 for e in AsmEdgeType}
_NODE_TYPE_MAP = {idx: nt for idx, nt in enumerate(CELL_TYPE_TO_NODE_TYPE)}


def _make_edge_index(src: list, tgt: list) -> torch.Tensor:
    """构建 edge_index 张量，空列表时返回 shape=(2,0) 的空张量"""
    if src:
        return torch.tensor([src, tgt], dtype=torch.long)
    return torch.empty((2, 0), dtype=torch.long)


def _build_rtl_structure(rtl: dict) -> dict:
    """构建 RTL 图结构张量（不含 coverage_label），返回 dict

    额外返回 _valid_edge_mask: list[bool]，标记原始 edges 中哪些边通过了
    node_id 过滤，用于标签提取时保持一致性。
    """
    nodes = rtl["nodes"]
    node_id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}

    node_cell_type = [get_cell_type_index(n["cell_type"]) for n in nodes]
    node_width = [n["width"] for n in nodes]

    src_list, tgt_list = [], []
    etype_list, ewidth_list = [], []
    esrc_port_list, etgt_port_list = [], []
    valid_edge_mask: list[bool] = []

    for e in rtl["edges"]:
        si = node_id_to_idx.get(e["source"])
        ti = node_id_to_idx.get(e["target"])
        if si is None or ti is None:
            valid_edge_mask.append(False)
            continue
        valid_edge_mask.append(True)
        src_list.append(si)
        tgt_list.append(ti)
        etype_list.append(_EDGE_TYPE_MAP.get(e["type"], 0))
        ewidth_list.append(e["width"])
        esrc_port_list.append(e["source_port_idx"])
        etgt_port_list.append(e["target_port_idx"])

    return {
        "node_cell_type": torch.tensor(node_cell_type, dtype=torch.long),
        "node_type": torch.tensor([_NODE_TYPE_MAP[ct] for ct in node_cell_type], dtype=torch.long),
        "node_width": torch.tensor(node_width, dtype=torch.long),
        "edge_index": _make_edge_index(src_list, tgt_list),
        "edge_type": torch.tensor(etype_list, dtype=torch.long),
        "edge_width": torch.tensor(ewidth_list, dtype=torch.long),
        "edge_source_port_idx": torch.tensor(esrc_port_list, dtype=torch.long),
        "edge_target_port_idx": torch.tensor(etgt_port_list, dtype=torch.long),
        "_valid_edge_mask": valid_edge_mask,
    }


def _build_asm_graph(
    asm_json_path: str, asm_encoding: Optional[torch.Tensor]
) -> dict:
    """构建 ASM 图张量，返回 dict"""
    with open(asm_json_path, encoding="utf-8") as f:
        asm = json.load(f)

    asm_node_ids = list(asm["nodes"].keys())
    asm_id_to_idx = {nid: i for i, nid in enumerate(asm_node_ids)}

    asm_nt = [
        _ASM_NODE_TYPE_MAP.get(
            asm["nodes"][nid]["node_type"],
            AsmNodeType.UNKNOWN.value - 1,
        )
        for nid in asm_node_ids
    ]

    asm_src, asm_tgt, asm_et = [], [], []
    for ae in asm["edges"]:
        s = asm_id_to_idx.get(ae["source"])
        t = asm_id_to_idx.get(ae["target"])
        if s is None or t is None:
            continue
        asm_src.append(s)
        asm_tgt.append(t)
        asm_et.append(_ASM_EDGE_TYPE_MAP.get(ae["edge_type"], 0))

    # 使用预计算的编码，或回退到零向量
    if asm_encoding is not None:
        asm_instr_enc = asm_encoding
    else:
        asm_instr_enc = torch.zeros(len(asm_node_ids), 256)

    return {
        "asm_node_type": torch.tensor(asm_nt, dtype=torch.long),
        "asm_instruction_encoding": asm_instr_enc,
        "asm_edge_index": _make_edge_index(asm_src, asm_tgt),
        "asm_edge_type": torch.tensor(asm_et, dtype=torch.long),
    }


def _read_dataset_index(dataset_dir: Path) -> list[tuple[str, str]]:
    """读取 dataset_index.jsonlines，返回 [(rtl_abs, asm_abs), ...]"""
    index_file = dataset_dir / "dataset_index.jsonlines"
    results: list[tuple[str, str]] = []
    with open(index_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            rtl = str(dataset_dir / entry["rtl"])
            asm = str(dataset_dir / entry["asm"])
            results.append((rtl, asm))
    return results


def _build_and_save_labels(
    args: tuple[str, str, str, str, int],
) -> bool:
    """阶段 2 worker: 仅提取标签并保存轻量 data_{idx}.pt

    RTL 图结构和 ASM 图已去重保存，此处仅保存 edge_labels + y + 引用文件名。
    通过 _valid_edge_mask 保证标签与图结构的边过滤一致。
    """
    rtl_json_path, rtl_filename, asm_filename, processed_dir, idx = args

    try:
        with open(rtl_json_path, encoding="utf-8") as f:
            rtl = json.load(f)

        nodes = rtl["nodes"]
        node_id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}

        elabel_list = []
        for e in rtl["edges"]:
            si = node_id_to_idx.get(e["source"])
            ti = node_id_to_idx.get(e["target"])
            if si is None or ti is None:
                continue
            elabel_list.append(e["coverage_label"])

        y = rtl.get("branch", 0.0) / 100.0

        data = DualGraphData(
            edge_labels=torch.tensor(elabel_list, dtype=torch.long),
            y=torch.tensor([[y]], dtype=torch.float),
            _rtl_file=rtl_filename,
            _asm_file=asm_filename,
        )

        pt_path = Path(processed_dir) / f"data_{idx}.pt"
        torch.save(data, pt_path)
        return True

    except Exception as e:
        logger.error("构建/保存标签失败 [%s]: %s", rtl_json_path, e)
        return False


def _compute_asm_cache_key(asm_path: str, config_fingerprint: str) -> str:
    h = hashlib.sha256()
    with open(asm_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    h.update(config_fingerprint.encode("utf-8"))
    return h.hexdigest()[:32]


class DualGraphDataset(Dataset):
    """
    双图数据集 - 继承自 PyG Dataset

    从 coverage-report-extractor 生成的数据集目录加载数据。
    目录须包含 dataset_index.jsonlines 索引文件。

    Args:
        root: 数据集根目录，包含 processed/ 子目录（PyG 约定）
        dataset_dir: coverage-report-extractor 输出目录（含 dataset_index.jsonlines）
        text_encoder_config: 文本编码器配置，None 时回退到零向量
        transform: 每次获取数据时应用的变换
        pre_transform: 处理前应用的变换（保存到磁盘）
        pre_filter: 处理前的过滤函数
    """

    def __init__(
        self,
        root: Optional[str] = "dataset_root",
        dataset_dir: Optional[Union[str, Path]] = None,
        text_encoder_config: Optional["TextEncoderConfig"] = None,
        num_workers: int = 0,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.root = root
        self._dataset_dir = Path(dataset_dir) if dataset_dir else None
        self._num_samples: Optional[int] = None
        self._idx_to_file: list[str] = []
        self._text_encoder_config = text_encoder_config
        self._num_workers = num_workers or min(os.cpu_count() or 1, 32)
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> List[str]:
        if self._dataset_dir is None:
            return []
        return [str(self._dataset_dir / "dataset_index.jsonlines")]

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
        if self._dataset_dir is None:
            logger.warning("未指定 dataset_dir，跳过处理")
            self._num_samples = 0
            return

        index_file = self._dataset_dir / "dataset_index.jsonlines"
        if not index_file.exists():
            logger.error("dataset_index.jsonlines 不存在: %s", index_file)
            self._num_samples = 0
            return

        flat_results = _read_dataset_index(self._dataset_dir)
        if not flat_results:
            logger.warning("dataset_index.jsonlines 为空")
            self._num_samples = 0
            return

        logger.info("读取索引完成: %d 个 (rtl, asm) 对", len(flat_results))

        # ── 阶段 1: 批量 GPU 编码（CodeBERT）+ 磁盘缓存 ──
        asm_encodings: dict[str, "torch.Tensor"] = {}
        unique_asm_paths = list({asm for _, asm in flat_results})

        if self._text_encoder_config is not None and unique_asm_paths:
            cache_dir = Path(self.root) / "asm_encoding_cache"
            config_fp = self._text_encoder_config.cache_fingerprint

            text_encoder = None

            def _get_encoder(*, projection_only: bool = False):
                nonlocal text_encoder
                if text_encoder is None:
                    from text_encoder import MultiGPUInstructionEncoder
                    text_encoder = MultiGPUInstructionEncoder(
                        self._text_encoder_config,
                        projection_only=projection_only,
                    )
                return text_encoder

            uncached_paths: list[str] = []

            for asm_path in tqdm(unique_asm_paths, desc="加载 ASM 编码缓存"):
                cache_key = _compute_asm_cache_key(asm_path, config_fp)
                cache_file = cache_dir / f"{cache_key}.pt"
                if cache_file.exists():
                    try:
                        pooled = torch.load(cache_file, weights_only=True)
                        asm_encodings[asm_path] = _get_encoder(
                            projection_only=True
                        ).project(pooled)
                        del pooled
                        continue
                    except Exception as e:
                        logger.warning("缓存文件损坏，将重新编码: %s (%s)", cache_file, e)
                uncached_paths.append(asm_path)

            logger.info(
                "阶段 1/2: ASM 编码 — %d 缓存命中, %d 待编码",
                len(asm_encodings),
                len(uncached_paths),
            )

            if uncached_paths:
                if text_encoder is not None and text_encoder._projection_only:
                    text_encoder = None
                encoder = _get_encoder(projection_only=False)
                cache_dir.mkdir(parents=True, exist_ok=True)

                new_count = 0
                for path, pooled_tensor in encoder.encode_asm_jsons_pooled_iter(
                    uncached_paths
                ):
                    cache_key = _compute_asm_cache_key(path, config_fp)
                    torch.save(pooled_tensor, cache_dir / f"{cache_key}.pt")
                    asm_encodings[path] = encoder.project(pooled_tensor)
                    del pooled_tensor
                    new_count += 1

                logger.info("编码完成，已缓存 %d 个新文件", new_count)

            logger.info("阶段 1 完成: %d 个编码结果", len(asm_encodings))
        else:
            logger.info("阶段 1/2: 跳过 GPU 编码（无编码器配置或无 ASM 文件）")

        # ── 阶段 1.5: 构建并保存去重 RTL 图 + ASM 图 ──
        processed_dir = self.processed_dir

        # --- RTL 图去重 ---
        rtl_module_to_path: dict[str, str] = {}
        for rtl_path, _ in flat_results:
            module_name = Path(rtl_path).stem
            if module_name not in rtl_module_to_path:
                rtl_module_to_path[module_name] = rtl_path

        rtl_module_to_file: dict[str, str] = {}
        for module_name, rtl_path in rtl_module_to_path.items():
            rtl_filename = f"rtl_{module_name}.pt"
            rtl_file_path = Path(processed_dir) / rtl_filename
            if not rtl_file_path.exists():
                with open(rtl_path, encoding="utf-8") as f:
                    rtl_json = json.load(f)
                rtl_data = _build_rtl_structure(rtl_json)
                rtl_data.pop("_valid_edge_mask", None)
                torch.save(rtl_data, rtl_file_path)
            rtl_module_to_file[module_name] = rtl_filename

        logger.info("阶段 1.5a: 保存 %d 个去重 RTL 图结构", len(rtl_module_to_file))

        # --- ASM 图去重 ---
        asm_path_to_file: dict[str, str] = {}
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
        save_args: list[tuple[str, str, str, str, int]] = []

        for i, (rtl_path, asm_path) in enumerate(flat_results):
            module_name = Path(rtl_path).stem
            rtl_filename = rtl_module_to_file.get(module_name)
            asm_filename = asm_path_to_file.get(asm_path)
            if rtl_filename is None or asm_filename is None:
                logger.error("图文件缺失，跳过: rtl=%s asm=%s", rtl_path, asm_path)
                continue
            save_args.append((rtl_path, rtl_filename, asm_filename, processed_dir, i))

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
        data = torch.load(
            Path(self.processed_dir) / filename, weights_only=False
        )

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


class RtlSizeBucketSampler(torch.utils.data.Sampler):
    """
    按 RTL 图节点数分桶的动态 batch sampler。

    原理：
    - 预扫描每个样本的 N_rtl（节点数），按对数桶分组
    - 桶内按 total_nodes ≤ token_budget 装箱，输出 List[List[int]]
    - 避免 batch 内出现 N_rtl 差异极大的样本，减少 padding 浪费

    与 follow_batch=["asm_node_type"] 兼容（PyGDataLoader 的 batch_sampler 路径）。
    """

    _BUCKET_EDGES = (128, 512, 2048)  # 按节点数划分的桶边界

    def __init__(
        self,
        dataset,
        token_budget: int = 8192,
        shuffle: bool = True,
        seed: int = 0,
    ):
        """
        Args:
            dataset:      DualGraphDataset 实例
            token_budget: 每个 batch 的最大节点总数
            shuffle:      是否在桶内随机打散
            seed:         随机种子
        """
        super().__init__()
        self.token_budget = token_budget
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

        # 预扫描：获取每个样本的 RTL 节点数
        sizes = self._scan_sizes(dataset)
        # 按桶分组
        buckets: dict[int, list[int]] = {k: [] for k in range(len(self._BUCKET_EDGES) + 1)}
        for idx, n in enumerate(sizes):
            bkt = sum(n >= e for e in self._BUCKET_EDGES)
            buckets[bkt].append(idx)
        self._buckets = [v for v in buckets.values() if v]
        self._sizes = sizes

    @staticmethod
    def _scan_sizes(dataset) -> list[int]:
        sizes = []
        for i in range(len(dataset)):
            data = dataset[i]
            # node_cell_type 是 RTL 节点的可靠代理字段
            n = int(data.node_cell_type.size(0)) if hasattr(data, "node_cell_type") else 1
            sizes.append(max(n, 1))
        return sizes

    def _make_batches(self) -> list[list[int]]:
        rng = torch.Generator()
        rng.manual_seed(self.seed + self._epoch)
        batches = []
        for bucket in self._buckets:
            indices = list(bucket)
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=rng).tolist()
                indices = [indices[p] for p in perm]
            batch, total = [], 0
            for idx in indices:
                n = self._sizes[idx]
                if batch and total + n > self.token_budget:
                    batches.append(batch)
                    batch, total = [], 0
                batch.append(idx)
                total += n
            if batch:
                batches.append(batch)
        if self.shuffle:
            perm = torch.randperm(len(batches), generator=rng).tolist()
            batches = [batches[p] for p in perm]
        return batches

    def __iter__(self):
        self._cached_batches = self._make_batches()
        yield from self._cached_batches

    def __len__(self) -> int:
        if not hasattr(self, "_cached_batches"):
            self._cached_batches = self._make_batches()
        return len(self._cached_batches)

    def set_epoch(self, epoch: int):
        self._epoch = epoch
        self._cached_batches = self._make_batches()


class DualGraphDataModule(L.LightningDataModule):
    """双图数据加载模块"""

    def __init__(
        self,
        root: str,
        dataset_dir: Optional[Union[str, Path]] = None,
        text_encoder_config: Optional["TextEncoderConfig"] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        transform: Optional[Callable] = None,
        use_bucketing: bool = False,
        token_budget: int = 8192,
    ):
        """
        Args:
            root: 数据集根目录（存放 processed/ .pt 文件）
            dataset_dir: coverage-report-extractor 输出目录（含 dataset_index.jsonlines）
            text_encoder_config: 文本编码器配置，None 时回退到零向量
            batch_size: 固定 batch 大小（use_bucketing=False 时生效）
            num_workers: 数据加载线程数
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
                    self.train_dataset, [train_len, val_len],
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
