#!/usr/bin/env python3
"""数据加载模块"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Callable

import torch
import lightning as L
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader

from cdfg_asm import AsmCDFGExtractor
from cdfg_rtl.data_types import EdgeType, get_cell_type_index
from cdfg_asm.data_types import AsmNodeType, AsmEdgeType
from tools import YosysRunner
from .data_types import DualGraphData

logger = logging.getLogger(__name__)

# 枚举名 → 0-indexed 映射表
_EDGE_TYPE_MAP = {e.name: e.value - 1 for e in EdgeType}
_ASM_NODE_TYPE_MAP = {e.name: e.value - 1 for e in AsmNodeType}
_ASM_EDGE_TYPE_MAP = {e.name: e.value - 1 for e in AsmEdgeType}


class DualGraphDataset(Dataset):
    """
    双图数据集 - 继承自 PyG Dataset

    支持磁盘持久化，数据按索引存储在 processed_dir 中。

    Args:
        root: 数据集根目录，包含 raw/ 和 processed/ 子目录
        data_list: DualGraphData 对象列表（首次运行时提供）
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
        >>> data_list = [DualGraphData(...), DualGraphData(...)]
        >>> dataset = DualGraphDataset(root='./data', data_list=data_list)

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
        data_list: Optional[List[DualGraphData]] = None,
        sim_results_dirs: Optional[List[str]] = None,
        module_names: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        """
        Args:
            root: 数据集根目录
            data_list: DualGraphData 对象列表（首次运行时提供）
            sim_results_dirs: simulation_results 目录路径列表
            module_names: 要标注的模块名列表（不含 .json 后缀），None 表示全部
            transform: 每次获取数据时应用的变换
            pre_transform: 处理前应用的变换
            pre_filter: 处理前的过滤函数
        """
        self.root = root
        self._data_list = data_list
        self._sim_results_dirs = sim_results_dirs or [
            "designs/cv32e40p/simulation_results"
        ]
        self._module_names = module_names or [
            "cv32e40p_int_controller",
        ]
        self._num_samples: Optional[int] = None
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> List[str]:
        return self._sim_results_dirs

    @property
    def processed_file_names(self) -> List[str]:
        """处理后的文件列表"""
        # 动态读取 processed_dir 中的文件
        if self._num_samples is not None:
            return [f"data_{i}.pt" for i in range(self._num_samples)]

        # 检查已存在的文件
        processed_path = Path(self.processed_dir)
        if processed_path.exists():
            files = sorted(
                f.name
                for f in processed_path.iterdir()
                if f.name.startswith("data_") and f.name.endswith(".pt")
            )
            if files:
                self._num_samples = len(files)
                return files

        # 如果提供了 data_list，返回预期文件名
        if self._data_list is not None:
            self._num_samples = len(self._data_list)
            return [f"data_{i}.pt" for i in range(self._num_samples)]

        return []

    def download(self):
        pass

    def process(self):
        """处理原始数据：遍历 simulation_results 生成标注 JSON 并转换为 .pt"""
        from annotation.data_annotate import DataAnnotator

        idx = 0

        for raw_path in self.raw_file_names:
            sim_results_dir = Path(raw_path)
            if not sim_results_dir.is_dir():
                logger.warning("simulation_results 目录不存在: %s", sim_results_dir)
                continue

            # 路径推导: designs/cv32e40p/simulation_results -> designs/cv32e40p
            design_dir = sim_results_dir.parent
            rtlil_json_dir = design_dir / "RTLIL_json"

            if not rtlil_json_dir.is_dir():
                try:
                    rtlil_json_dir.mkdir(parents=True, exist_ok=True)
                    logger.info("已创建 RTLIL JSON 目录: %s", rtlil_json_dir)
                except Exception as e:
                    raise RuntimeError(f"无法创建 RTLIL JSON 目录: {rtlil_json_dir}") from e

            for test_dir in sorted(sim_results_dir.iterdir()):
                if not test_dir.is_dir():
                    continue

                # 1. 生成标注 RTL JSON
                rtl_files = self._generate_annotated_json(
                    rtlil_json_dir, test_dir, DataAnnotator
                )

                # 2. 生成 ASM CDFG JSON
                asm_json = self._generate_asm_cdfg_json(test_dir)
                if asm_json:
                    logger.info("  ASM CDFG: %s", Path(asm_json).name)

                # 3. 构建 DualGraphData 并保存 .pt
                for rtl_file in rtl_files:
                    try:
                        data = self._build_dual_graph_data(rtl_file, asm_json)
                        if data is not None:
                            pt_path = Path(self.processed_dir) / f"data_{idx}.pt"
                            torch.save(data, pt_path)
                            idx += 1
                    except Exception as e:
                        logger.error("构建 DualGraphData 失败 [%s]: %s", rtl_file, e)

        self._num_samples = idx
        logger.info("数据处理完成，共 %d 个样本", self._num_samples)

    def _generate_annotated_json(
        self,
        rtlil_json_dir: Path,
        test_dir: Path,
        annotator_cls: type,
    ) -> List[str]:
        """
        为单个测试生成所有模块的标注 JSON

        Args:
            rtlil_json_dir: RTLIL JSON 文件目录
            test_dir: 测试目录 (simulation_results/{test_name}/)
            annotator_cls: DataAnnotator 类引用

        Returns:
            生成的文件路径列表
        """
        coverage_report_dir = test_dir / "coverage" / "report"
        if not coverage_report_dir.is_dir():
            return []

        output_dir = test_dir / "annotated"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 收集 RTLIL JSON 文件（按 module_names 过滤）
        if self._module_names is not None:
            # 检查缺失的 JSON 文件并尝试通过 YosysRunner 生成
            design_dir = rtlil_json_dir.parent
            flist = design_dir / f"{design_dir.name}.flist"

            for name in self._module_names:
                json_path = rtlil_json_dir / f"{name}.json"
                if not json_path.exists():
                    logger.info("RTLIL JSON 不存在，尝试生成: %s", json_path)
                    try:
                        runner = YosysRunner()
                        runner.get_json(
                            top_module=name,
                            output_dir=rtlil_json_dir,
                            flist=flist,
                            files=None,
                        )
                    except Exception as e:
                        logger.error("YosysRunner 生成失败 [%s]: %s", name, e)

            json_files = [
                rtlil_json_dir / f"{name}.json"
                for name in self._module_names
                if (rtlil_json_dir / f"{name}.json").exists()
            ]
        else:
            json_files = sorted(rtlil_json_dir.glob("*.json"))

        generated: List[str] = []

        for json_file in json_files:
            module_name = json_file.stem
            output_path = output_dir / f"{module_name}.json"

            # 跳过已标注的文件
            if output_path.exists():
                generated.append(str(output_path))
                continue

            try:
                annotator = annotator_cls.from_args(
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
                logger.error(
                    "标注失败 [%s/%s]: %s", test_dir.name, module_name, e
                )

        return generated

    def _generate_asm_cdfg_json(self, test_dir: Path) -> Optional[str]:
        """
        为单个测试生成 ASM CDFG JSON

        在 test_dir 中查找 {test_name}.S 文件，提取 ASM CDFG 并导出为
        {test_name}.json（与 .S 文件同目录同名）。

        Args:
            test_dir: 测试目录 (simulation_results/{test_name}/)

        Returns:
            生成的 JSON 文件路径，失败返回 None
        """
        test_name = test_dir.name
        s_file = test_dir / f"{test_name}.S"

        if not s_file.exists():
            logger.debug("未找到汇编文件: %s", s_file)
            return None

        output_path = test_dir / f"{test_name}.json"

        try:
            extractor = AsmCDFGExtractor(verbose=False)
            cdfg = extractor.extract_from_file(str(s_file))

            # JSON 序列化
            export_data = cdfg.to_dict()

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info("ASM CDFG 已生成: %s", output_path)
            return str(output_path)

        except Exception as e:
            logger.error("ASM CDFG 生成失败 [%s]: %s", test_name, e)
            return None

    def _build_dual_graph_data(
        self, rtl_json_path: str, asm_json_path: str
    ) -> Optional[DualGraphData]:
        """
        从标注 RTL JSON 和 ASM CDFG JSON 构建 DualGraphData 张量

        Args:
            rtl_json_path: 标注后的 RTL CDFG JSON 路径
            asm_json_path: ASM CDFG JSON 路径

        Returns:
            DualGraphData 对象，失败返回 None
        """
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
                continue  # 跳过悬空边
            src_list.append(si)
            tgt_list.append(ti)
            etype_list.append(_EDGE_TYPE_MAP.get(e["type"], 0))
            ewidth_list.append(e["width"])
            esrc_port_list.append(e["source_port_idx"])
            etgt_port_list.append(e["target_port_idx"])
            elabel_list.append(e["coverage_label"])

        if src_list:
            edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # ── ASM 图 ──
        if asm_json_path is not None:
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
            asm_instr_enc = torch.zeros(len(asm_node_ids), 256)  # 占位
            if asm_src:
                asm_edge_index = torch.tensor(
                    [asm_src, asm_tgt], dtype=torch.long
                )
            else:
                asm_edge_index = torch.empty((2, 0), dtype=torch.long)
            asm_edge_type = torch.tensor(asm_et, dtype=torch.long)
        else:
            raise RuntimeError(f"ASM JSON 文件缺失，无法构建DualGraphData图: {asm_json_path}")
            # 无 ASM 文件 → 空图
            # asm_node_type = torch.empty(0, dtype=torch.long)
            # asm_instr_enc = torch.empty((0, 256))
            # asm_edge_index = torch.empty((2, 0), dtype=torch.long)
            # asm_edge_type = torch.empty(0, dtype=torch.long)

        # ── 图级标签 y ──
        y = rtl.get("branch_coverage", 0.0) / 100.0  # 百分比 → [0, 1] 归一化

        return DualGraphData(
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

    def len(self) -> int:
        """返回数据集大小"""
        if self._num_samples is not None:
            return self._num_samples

        # 从磁盘统计文件数量
        processed_path = Path(self.processed_dir)
        if processed_path.exists():
            files = [
                f.name
                for f in processed_path.iterdir()
                if f.name.startswith("data_") and f.name.endswith(".pt")
            ]
            self._num_samples = len(files)
            return self._num_samples

        return 0

    def get(self, idx: int) -> DualGraphData:
        """从磁盘加载指定索引的数据"""
        data = torch.load(
            Path(self.processed_dir) / f"data_{idx}.pt", weights_only=False
        )
        return data


class DualGraphDataModule(L.LightningDataModule):
    """双图数据加载模块"""

    def __init__(
        self,
        root: str,
        train_data: Optional[List[DualGraphData]] = None,
        val_data: Optional[List[DualGraphData]] = None,
        test_data: Optional[List[DualGraphData]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            root: 数据集根目录
            train_data: 训练数据列表（首次运行时提供）
            val_data: 验证数据列表（首次运行时提供）
            test_data: 测试数据列表（首次运行时提供）
            batch_size: 批大小
            num_workers: 数据加载线程数
            transform: 数据变换
        """
        super().__init__()
        self.root = root
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """初始化数据集"""
        root_path = Path(self.root)
        if stage == "fit" or stage is None:
            self.train_dataset = DualGraphDataset(
                root=str(root_path / "train"),
                data_list=self.train_data,
                transform=self.transform,
            )
            if self.val_data is not None or (root_path / "val" / "processed").exists():
                self.val_dataset = DualGraphDataset(
                    root=str(root_path / "val"),
                    data_list=self.val_data,
                    transform=self.transform,
                )

        if stage == "test" or stage is None:
            if self.test_data is not None or (root_path / "test" / "processed").exists():
                self.test_dataset = DualGraphDataset(
                    root=str(root_path / "test"),
                    data_list=self.test_data,
                    transform=self.transform,
                )

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
