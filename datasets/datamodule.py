#!/usr/bin/env python3
"""数据加载模块"""

import hashlib
import json

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

from cdfg_asm import AsmCDFGExtractor
from cdfg_rtl.data_types import EdgeType, get_cell_type_index
from cdfg_asm.data_types import AsmNodeType, AsmEdgeType
from scripts.RTLIL_extract import extract_rtlil_json
from .data_types import DualGraphData

logger = logging.getLogger(__name__)

# 枚举名 → 0-indexed 映射表
_EDGE_TYPE_MAP = {e.name: e.value - 1 for e in EdgeType}
_ASM_NODE_TYPE_MAP = {e.name: e.value - 1 for e in AsmNodeType}
_ASM_EDGE_TYPE_MAP = {e.name: e.value - 1 for e in AsmEdgeType}


def _make_edge_index(src: list, tgt: list) -> torch.Tensor:
    """构建 edge_index 张量，空列表时返回 shape=(2,0) 的空张量"""
    if src:
        return torch.tensor([src, tgt], dtype=torch.long)
    return torch.empty((2, 0), dtype=torch.long)


def _generate_annotated_json(
    rtlil_json_dir: Path,
    test_dir: Path,
    module_names: Optional[list[str]],
) -> list[str]:
    """为单个测试生成所有模块的标注 JSON（模块级函数，供多进程调用）

    两遍扫描策略：
    1. 第一遍：检查已存在的标注文件，直接收集路径
    2. 第二遍：仅当存在缺失文件时，才检查覆盖率目录并执行标注
    """
    output_dir = test_dir / "annotated"

    # 收集 RTLIL JSON 文件（按 module_names 过滤）
    if module_names is not None:
        json_files = [
            p
            for name in module_names
            if (p := rtlil_json_dir / f"{name}.json").exists()
        ]
    else:
        json_files = sorted(rtlil_json_dir.glob("*.json"))

    # ── 第一遍：收集已存在的标注文件，记录缺失项 ──
    generated: list[str] = []
    need_annotate: list[tuple[Path, Path]] = []

    for json_file in json_files:
        output_path = output_dir / f"{json_file.stem}.json"

        if output_path.exists():
            generated.append(str(output_path))
        else:
            need_annotate.append((json_file, output_path))

    # ── 第二遍：仅当存在缺失文件时执行标注 ──
    if not need_annotate:
        return generated

    coverage_report_dir = test_dir / "coverage" / "report"
    if not coverage_report_dir.is_dir():
        logger.warning(
            "覆盖率目录不存在 [%s]，跳过 %d 个待标注模块（已收集 %d 个已有文件）",
            coverage_report_dir,
            len(need_annotate),
            len(generated),
        )
        return generated

    from annotation.data_annotate import DataAnnotator

    output_dir.mkdir(parents=True, exist_ok=True)

    for json_file, output_path in need_annotate:
        try:
            annotator = DataAnnotator.from_args(
                design_json=str(json_file),
                coverage_dir=str(coverage_report_dir),
                propagate=False,
            )
            annotator.extract_cdfg()
            annotator.find_coverage_files()
            annotator.annotate()
            annotator.export_json(str(output_path))
            generated.append(str(output_path))
        except Exception as e:
            logger.error("标注失败 [%s/%s]: %s", test_dir.name, json_file.stem, e)

    return generated


def _generate_asm_cdfg_json(test_dir: Path) -> Optional[str]:
    """为单个测试生成 ASM CDFG JSON（模块级函数，供多进程调用）"""
    test_name = test_dir.name
    s_file = test_dir / f"{test_name}.S"

    if not s_file.exists():
        logger.debug("未找到汇编文件: %s", s_file)
        return None

    output_path = test_dir / f"{test_name}.json"

    # 跳过已生成的 ASM CDFG JSON（验证完整性）
    if output_path.exists():
        try:
            with open(output_path, "rb") as f:
                orjson.loads(f.read())
            return str(output_path)
        except Exception:
            logger.warning("ASM CDFG JSON 损坏，将重新生成: %s", output_path)
            output_path.unlink()

    try:
        extractor = AsmCDFGExtractor(verbose=False)
        cdfg = extractor.extract_from_file(str(s_file))
        export_data = cdfg.to_dict()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        # 写后验证：确保生成的 JSON 可正常解析
        with open(output_path, "rb") as f:
            orjson.loads(f.read())

        logger.info("ASM CDFG 已生成: %s", output_path)
        return str(output_path)

    except Exception as e:
        logger.error("ASM CDFG 生成失败 [%s]: %s", test_name, e)
        return None


def _process_single_test(
    args: tuple[str, str, Optional[list[str]]],
) -> list[tuple[str, Optional[str]]]:
    """阶段 1 worker: 单个测试目录的 CPU 预处理（RTL 标注 + ASM 提取）

    Args:
        args: (rtlil_json_dir, test_dir, module_names)

    Returns:
        [(rtl_json_path, asm_json_path), ...] 列表
    """
    rtlil_json_dir_str, test_dir_str, module_names = args
    rtlil_json_dir = Path(rtlil_json_dir_str)
    test_dir = Path(test_dir_str)

    # RTL 标注
    rtl_files = _generate_annotated_json(rtlil_json_dir, test_dir, module_names)

    # ASM CDFG 提取
    asm_json = _generate_asm_cdfg_json(test_dir)

    return [(rtl, asm_json) for rtl in rtl_files]


# ── 阶段 3 worker 共享内存支持 ──
_save_worker_packed: Optional["torch.Tensor"] = None  # 合并后的共享内存 tensor
_save_worker_slices: dict[str, tuple[int, int]] = {}  # path → (offset, count)


def _init_save_worker(
    packed: "torch.Tensor", slices: dict[str, tuple[int, int]]
) -> None:
    """Pool initializer: 注入合并的共享内存 tensor 和偏移索引"""
    global _save_worker_packed, _save_worker_slices
    _save_worker_packed = packed
    _save_worker_slices = slices


def _build_and_save(
    args: tuple[str, Optional[str], str, int],
) -> bool:
    """阶段 3 worker: 构建 DualGraphData 并保存 .pt

    通过 _save_worker_packed 共享内存 tensor 和偏移索引获取编码，
    避免 pickle 序列化导致的 tensor 深拷贝和内存放大。

    Args:
        args: (rtl_json_path, asm_json_path, processed_dir, idx)

    Returns:
        True 成功, False 失败
    """
    rtl_json_path, asm_json_path, processed_dir, idx = args

    # ASM 缺失提前返回，避免白费 RTL 解析开销
    if asm_json_path is None:
        logger.error("ASM JSON 文件缺失，跳过构建: rtl=%s", rtl_json_path)
        return False

    # 从共享内存合并 tensor 中按偏移切片
    slice_info = _save_worker_slices.get(asm_json_path)
    asm_encoding = _save_worker_packed[slice_info[0] : slice_info[0] + slice_info[1]] if slice_info else None

    try:
        # ── RTL 图 ──
        with open(rtl_json_path, encoding="utf-8") as f:
            rtl = json.load(f)

        nodes = rtl["nodes"]
        node_id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}

        node_cell_type = [get_cell_type_index(n["cell_type"]) for n in nodes]
        node_width = [n["width"] for n in nodes]

        src_list, tgt_list = [], []
        etype_list, ewidth_list = [], []
        esrc_port_list, etgt_port_list = [], []
        elabel_list = []

        for e in rtl["edges"]:
            si = node_id_to_idx.get(e["source"])
            ti = node_id_to_idx.get(e["target"])
            if si is None or ti is None:
                continue
            src_list.append(si)
            tgt_list.append(ti)
            etype_list.append(_EDGE_TYPE_MAP.get(e["type"], 0))
            ewidth_list.append(e["width"])
            esrc_port_list.append(e["source_port_idx"])
            etgt_port_list.append(e["target_port_idx"])
            elabel_list.append(e["coverage_label"])

        edge_index = _make_edge_index(src_list, tgt_list)

        # ── ASM 图 ──（asm_json_path 已保证非 None）
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

        asm_node_type = torch.tensor(asm_nt, dtype=torch.long)

        # 使用预计算的编码，或回退到零向量
        if asm_encoding is not None:
            asm_instr_enc = asm_encoding
        else:
            asm_instr_enc = torch.zeros(len(asm_node_ids), 256)

        asm_edge_index = _make_edge_index(asm_src, asm_tgt)
        asm_edge_type = torch.tensor(asm_et, dtype=torch.long)

        # ── 图级标签 y ──
        y = rtl.get("branch_coverage", 0.0) / 100.0

        data = DualGraphData(
            node_cell_type=torch.tensor(node_cell_type, dtype=torch.long),
            node_width=torch.tensor(node_width, dtype=torch.long),
            edge_index=edge_index,
            edge_type=torch.tensor(etype_list, dtype=torch.long),
            edge_width=torch.tensor(ewidth_list, dtype=torch.long),
            edge_source_port_idx=torch.tensor(esrc_port_list, dtype=torch.long),
            edge_target_port_idx=torch.tensor(etgt_port_list, dtype=torch.long),
            asm_node_type=asm_node_type,
            asm_instruction_encoding=asm_instr_enc,
            asm_edge_index=asm_edge_index,
            asm_edge_type=asm_edge_type,
            edge_labels=torch.tensor(elabel_list, dtype=torch.long),
            y=torch.tensor([[y]], dtype=torch.float),
        )

        pt_path = Path(processed_dir) / f"data_{idx}.pt"
        torch.save(data, pt_path)
        return True

    except Exception as e:
        logger.error("构建/保存 DualGraphData 失败 [%s]: %s", rtl_json_path, e)
        return False


def _compute_asm_cache_key(asm_path: str, config_fingerprint: str) -> str:
    """计算 ASM 文件的缓存键（基于文件内容 + 编码器配置指纹）

    Args:
        asm_path: ASM JSON 文件路径
        config_fingerprint: TextEncoderConfig.cache_fingerprint

    Returns:
        32 字符的十六进制缓存键
    """
    h = hashlib.sha256()
    with open(asm_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    h.update(config_fingerprint.encode("utf-8"))
    return h.hexdigest()[:32]


class DualGraphDataset(Dataset):
    """
    双图数据集 - 继承自 PyG Dataset

    支持磁盘持久化，数据按索引存储在 processed_dir 中。
    通过 sim_results_dirs 指定原始数据路径，自动执行标注和张量构建。

    Args:
        root: 数据集根目录，包含 raw/ 和 processed/ 子目录
        sim_results_dirs: simulation_results 目录路径列表
        module_names: 要标注的模块名列表（不含 .json 后缀），None 表示全部
        text_encoder_config: 文本编码器配置，None 时回退到零向量
        transform: 每次获取数据时应用的变换
        pre_transform: 处理前应用的变换（保存到磁盘）
        pre_filter: 处理前的过滤函数

    目录结构:
        root/
        ├── raw/              # 原始文件（可选）
        └── processed/        # 处理后的 .pt 文件
            ├── data_0.pt
            ├── data_1.pt
            └── ...

    Example:
        >>> # 首次创建数据集（处理并保存）
        >>> dataset = DualGraphDataset(
        ...     root='./data',
        ...     sim_results_dirs=['designs/cv32e40p/simulation_results'],
        ... )

        >>> # 后续加载（从磁盘读取）
        >>> dataset = DualGraphDataset(root='./data')
        >>> len(dataset)
        2
        >>> dataset[0]
        DualGraphData(...)
    """

    def __init__(
        self,
        root: Optional[str] = "dataset_root",
        sim_results_dirs: Optional[List[Union[str, Path]]] = None,
        module_names: Optional[List[str]] = None,
        text_encoder_config: Optional["TextEncoderConfig"] = None,
        num_workers: int = 0,
        preprocess_only: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        """
        Args:
            root: 数据集根目录
            sim_results_dirs: simulation_results 目录路径列表
            module_names: 要标注的模块名列表（不含 .json 后缀），None 表示全部
            text_encoder_config: 文本编码器配置，None 时回退到零向量
            num_workers: CPU 并行进程数，0 = 自动检测 min(cpu_count, 32)
            preprocess_only: 仅执行阶段 1 预处理（生成中间 JSON），跳过 GPU 编码和张量构建
            transform: 每次获取数据时应用的变换
            pre_transform: 处理前应用的变换
            pre_filter: 处理前的过滤函数
        """
        self.root = root
        self._sim_results_dirs = sim_results_dirs or [
            "designs/cv32e40p/simulation_results"
        ]
        self._module_names = module_names or [
            "cv32e40p_alu_div",
            "cv32e40p_ff_one",
            "cv32e40p_mult",
            "cv32e40p_register_file",
            "cv32e40p_controller",
            "cv32e40p_decoder",
            "cv32e40p_aligner",
            "cv32e40p_compressed_decoder",
            "cv32e40p_int_controller",
        ]
        self._num_samples: Optional[int] = None
        self._idx_to_file: list[str] = []  # 逻辑索引 → 实际文件名
        self._preprocess_only = preprocess_only
        self._text_encoder_config = text_encoder_config
        self._num_workers = num_workers or min(os.cpu_count() or 1, 32)
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> List[str]:
        return self._sim_results_dirs

    def _scan_processed_files(self) -> list[str]:
        """扫描 processed_dir 中的 data_*.pt 文件并更新缓存

        建立逻辑索引 → 实际文件名的映射，支持非连续索引。
        """
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
        """处理后的文件列表"""
        if self._idx_to_file:
            return self._idx_to_file
        return self._scan_processed_files()

    def download(self):
        pass

    def process(self):
        """三阶段流水线处理：多进程 CPU 预处理 → 批量 GPU 编码 → 多进程张量构建

        应通过 DualGraphDataModule.prepare_data() 在 DDP 初始化之前调用，
        避免 NCCL barrier 超时。
        """
        # 收集所有 (rtlil_json_dir, test_dir, module_names) 任务
        task_args: list[tuple[str, str, Optional[list[str]]]] = []

        for raw_path in self.raw_file_names:
            sim_results_dir = Path(raw_path)
            if not sim_results_dir.is_dir():
                logger.warning("simulation_results 目录不存在: %s", sim_results_dir)
                continue

            design_dir = sim_results_dir.parent
            rtlil_json_dir = design_dir / "RTLIL_json"

            # RTLIL JSON 预生成：仅当存在缺失的模块 JSON 时才调用
            if self._module_names is not None:
                missing = [
                    name
                    for name in self._module_names
                    if not (rtlil_json_dir / f"{name}.json").exists()
                ]
                if missing:
                    logger.info(
                        "RTLIL JSON 缺失 %d/%d 个模块，开始生成",
                        len(missing),
                        len(self._module_names),
                    )
                    extract_rtlil_json(sim_results_dir, missing)
                else:
                    logger.info("所有 RTLIL JSON 已存在，跳过生成: %s", rtlil_json_dir)
            else:
                extract_rtlil_json(sim_results_dir, self._module_names)

            for test_dir in sorted(sim_results_dir.iterdir()):
                if not test_dir.is_dir():
                    continue
                task_args.append(
                    (
                        str(rtlil_json_dir),
                        str(test_dir),
                        self._module_names,
                    )
                )

        if not task_args:
            logger.warning("未找到任何测试目录")
            self._num_samples = 0
            return

        # ── 阶段 1: 多进程 CPU 预处理（RTL 标注 + ASM 提取）──
        logger.info(
            "阶段 1/3: 多进程 CPU 预处理 (%d 个测试, %d workers)",
            len(task_args),
            self._num_workers,
        )
        with Pool(self._num_workers) as pool:
            all_results = pool.map(_process_single_test, task_args)

        # 展平结果: [(rtl_path, asm_path), ...]
        flat_results: list[tuple[str, Optional[str]]] = []
        for result_list in all_results:
            flat_results.extend(result_list)

        if not flat_results:
            logger.warning("阶段 1 未产生任何有效结果")
            self._num_samples = 0
            return

        logger.info("阶段 1 完成: %d 个 (rtl, asm) 对", len(flat_results))

        # preprocess_only 模式：仅执行阶段 1，跳过 GPU 编码和张量构建
        if self._preprocess_only:
            logger.info("preprocess_only 模式：跳过阶段 2/3，仅生成中间 JSON 文件")
            self._num_samples = 0
            return

        # ── 阶段 2: 批量 GPU 编码（CodeBERT）+ 磁盘缓存 ──
        asm_encodings: dict[str, "torch.Tensor"] = {}
        unique_asm_paths = list({asm for _, asm in flat_results if asm is not None})

        if self._text_encoder_config is not None and unique_asm_paths:
            cache_dir = Path(self.root) / "asm_encoding_cache"
            config_fp = self._text_encoder_config.cache_fingerprint

            # 延迟创建编码器：全缓存命中时仅加载投影层，节省显存
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

            # ── 2a: 串行加载缓存 + 即时投影 ──
            # 加载一个 768 维 tensor 后立即投影为 256 维并释放，
            # 避免全量 768 维 tensor 驻留内存 + 并发 torch.load 碎片化
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
                        del pooled  # 立即释放 768 维 tensor
                        continue
                    except Exception as e:
                        logger.warning(
                            "缓存文件损坏，将重新编码: %s (%s)", cache_file, e
                        )
                uncached_paths.append(asm_path)

            logger.info(
                "阶段 2/3: ASM 编码 — %d 缓存命中, %d 待编码",
                len(asm_encodings),
                len(uncached_paths),
            )

            # ── 2b: 编码未缓存项 + 即时投影 + 即时缓存 ──
            if uncached_paths:
                # 需要完整编码器（含 CodeBERT 模型）
                if text_encoder is not None and text_encoder._projection_only:
                    # 之前只加载了投影层，需要重建完整编码器
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
                    del pooled_tensor  # 立即释放 768 维 tensor
                    new_count += 1

                logger.info("编码完成，已缓存 %d 个新文件", new_count)

            logger.info("阶段 2 完成: %d 个编码结果", len(asm_encodings))
        else:
            logger.info("阶段 2/3: 跳过 GPU 编码（无编码器配置或无 ASM 文件）")

        # ── 阶段 3: 多进程张量构建 + 保存 ──
        processed_dir = self.processed_dir

        # 合并所有编码 tensor 为单个共享内存 tensor（仅 1 个 fd）
        if asm_encodings:
            packed_parts = []
            encoding_slices: dict[str, tuple[int, int]] = {}
            offset = 0
            for path, enc in asm_encodings.items():
                n = enc.shape[0]
                packed_parts.append(enc)
                encoding_slices[path] = (offset, n)
                offset += n
            packed_tensor = torch.cat(packed_parts, dim=0)
            packed_tensor.share_memory_()
            del asm_encodings, packed_parts  # 释放原始 tensor
        else:
            packed_tensor = torch.empty(0)
            encoding_slices = {}

        # save_args 不再包含 tensor，通过 initializer 传入共享内存
        save_args: list[tuple[str, Optional[str], str, int]] = []

        for i, (rtl_path, asm_path) in enumerate(flat_results):
            save_args.append((rtl_path, asm_path, processed_dir, i))

        logger.info(
            "阶段 3/3: 多进程张量构建 + 保存 (%d 个样本, %d workers)",
            len(save_args),
            self._num_workers,
        )
        with Pool(
            self._num_workers,
            initializer=_init_save_worker,
            initargs=(packed_tensor, encoding_slices),
        ) as pool:
            results = pool.map(_build_and_save, save_args)

        success_count = sum(1 for r in results if r)

        # 紧缩索引，消除失败样本导致的缺口
        actual_idx = 0
        for i, r in enumerate(results):
            if r and i != actual_idx:
                src = Path(processed_dir) / f"data_{i}.pt"
                dst = Path(processed_dir) / f"data_{actual_idx}.pt"
                src.rename(dst)
            if r:
                actual_idx += 1

        self._num_samples = success_count
        self._idx_to_file = [f"data_{i}.pt" for i in range(success_count)]
        logger.info("数据处理完成: %d/%d 个样本成功", success_count, len(save_args))

    def len(self) -> int:
        """返回数据集大小"""
        if self._num_samples is not None:
            return self._num_samples
        files = self._scan_processed_files()
        return len(files)

    def get(self, idx: int) -> DualGraphData:
        """从磁盘加载指定索引的数据"""
        if self._idx_to_file:
            filename = self._idx_to_file[idx]
        else:
            filename = f"data_{idx}.pt"
        data = torch.load(
            Path(self.processed_dir) / filename, weights_only=False
        )
        return data


class DualGraphDataModule(L.LightningDataModule):
    """双图数据加载模块"""

    def __init__(
        self,
        root: str,
        sim_results_dirs: Optional[List[str]] = None,
        module_names: Optional[List[str]] = None,
        text_encoder_config: Optional["TextEncoderConfig"] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        preprocess_only: bool = False,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            root: 数据集根目录
            sim_results_dirs: simulation_results 目录路径列表
            module_names: 要标注的模块名列表（不含 .json 后缀），None 表示全部
            text_encoder_config: 文本编码器配置，None 时回退到零向量
            batch_size: 批大小
            num_workers: 数据加载线程数
            preprocess_only: 仅执行阶段 1 预处理（生成中间 JSON），跳过 GPU 编码和张量构建
            transform: 数据变换
        """
        super().__init__()
        self.root = root
        self.sim_results_dirs = sim_results_dirs
        self.module_names = module_names
        self.text_encoder_config = text_encoder_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocess_only = preprocess_only
        self.transform = transform

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _make_dataset(self, split: str) -> DualGraphDataset:
        """创建指定 split 的 DualGraphDataset"""
        return DualGraphDataset(
            root=str(Path(self.root) / split),
            sim_results_dirs=self.sim_results_dirs,
            module_names=self.module_names,
            text_encoder_config=self.text_encoder_config,
            preprocess_only=self.preprocess_only,
            transform=self.transform,
        )

    def prepare_data(self):
        """在 DDP 初始化之前完成全部数据处理（仅 rank 0）

        PyTorch Lightning 保证 prepare_data() 仅在主进程调用，
        且在 DDP 进程组创建之前执行，因此不存在 NCCL barrier 超时问题。
        所有 GPU 编码和磁盘持久化在此阶段完成。
        """
        self._make_dataset("train")

    def setup(self, stage: Optional[str] = None):
        """初始化数据集

        val 集缺失时自动从 train 集按 9:1 划分，确保 Lightning sanity check 通过。
        """
        root_path = Path(self.root)
        if stage == "fit" or stage is None:
            self.train_dataset = self._make_dataset("train")
            if (root_path / "val" / "processed").exists():
                self.val_dataset = self._make_dataset("val")
            else:
                # 自动划分：90% train, 10% val
                from torch.utils.data import random_split
                full_len = len(self.train_dataset)
                val_len = max(1, int(full_len * 0.1))
                train_len = full_len - val_len
                self.train_dataset, self.val_dataset = random_split(
                    self.train_dataset, [train_len, val_len]
                )
                logger.info(
                    "val 集缺失，自动划分: train=%d, val=%d", train_len, val_len
                )

        if stage == "test" or stage is None:
            if (root_path / "test" / "processed").exists():
                self.test_dataset = self._make_dataset("test")

    def _create_loader(self, dataset, shuffle: bool) -> PyGDataLoader:
        """创建 PyG DataLoader"""
        return PyGDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            follow_batch=["asm_node_type"],  # 为 ASM 图生成 batch 索引
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
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
