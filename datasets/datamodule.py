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
from .data_types import DualGraphData

logger = logging.getLogger(__name__)


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
        """处理原始数据：遍历 simulation_results 生成标注 JSON"""
        from annotation.data_annotate import DataAnnotator

        generated_files: List[str] = []

        for raw_path in self.raw_file_names:
            sim_results_dir = Path(raw_path)
            if not sim_results_dir.is_dir():
                logger.warning("simulation_results 目录不存在: %s", sim_results_dir)
                continue

            # 路径推导: designs/cv32e40p/simulation_results -> designs/cv32e40p
            design_dir = sim_results_dir.parent
            rtlil_json_dir = design_dir / "RTLIL_json"

            if not rtlil_json_dir.is_dir():
                logger.warning("RTLIL_json 目录不存在: %s", rtlil_json_dir)
                continue

            for test_dir in sorted(sim_results_dir.iterdir()):
                if not test_dir.is_dir():
                    continue

                # RTL CDFG 标注
                files = self._generate_annotated_json(
                    rtlil_json_dir, test_dir, DataAnnotator
                )
                generated_files.extend(files)

                # ASM CDFG 生成
                asm_json = self._generate_asm_cdfg_json(test_dir)
                if asm_json:
                    logger.info("  ASM CDFG: %s", Path(asm_json).name)

        self._num_samples = len(generated_files)
        logger.info("标注 JSON 生成完成，共 %d 个文件", self._num_samples)

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

            # JSON 序列化（与 cdfg_asm/extractor.py:390-430 格式一致）
            export_data = {
                "module_name": cdfg.module_name,
                "source_file": cdfg.source_file,
                "entry_node": cdfg.entry_node,
                "exit_nodes": cdfg.exit_nodes,
                "total_instructions": cdfg.total_instructions,
                "total_basic_blocks": cdfg.total_basic_blocks,
                "nodes": {
                    node_id: {
                        "id": node.id,
                        "label": node.label,
                        "node_type": node.node_type.name,
                        "start_line": node.start_line,
                        "end_line": node.end_line,
                        "instr_count": node.instr_count,
                        "instructions": [
                            {"mnemonic": i.mnemonic, "operands": i.operands}
                            for i in node.instructions
                        ],
                        "defs": list(node.defs),
                        "uses": list(node.uses),
                        "successors": node.successors,
                        "predecessors": node.predecessors,
                        "is_loop_header": node.is_loop_header,
                        "branch_condition": node.branch_condition,
                        "branch_target": node.branch_target,
                    }
                    for node_id, node in cdfg.nodes.items()
                },
                "edges": [
                    {
                        "source": e.source,
                        "target": e.target,
                        "edge_type": e.edge_type.name,
                        "register": e.register,
                        "condition": e.condition,
                        "is_back_edge": e.is_back_edge,
                    }
                    for e in cdfg.edges
                ],
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info("ASM CDFG 已生成: %s", output_path)
            return str(output_path)

        except Exception as e:
            logger.error("ASM CDFG 生成失败 [%s]: %s", test_name, e)
            return None

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
