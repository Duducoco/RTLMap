#!/usr/bin/env python3
"""数据加载模块"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
import lightning as L

from models.data_types import DualGraphData


class DualGraphDataset(Dataset):
    """双图数据集封装"""

    def __init__(self, data_list: List[DualGraphData]):
        self.data_list = data_list

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> DualGraphData:
        return self.data_list[idx]


def collate_dual_graph(batch: List[DualGraphData]) -> DualGraphData:
    """
    将多个 DualGraphData 合并为一个 batch

    处理逻辑：
    1. 拼接节点特征，生成 batch 索引
    2. 拼接边索引，偏移节点 ID
    3. 拼接边标签、边类型等
    """
    # RTL 图
    rtl_x_list = []
    rtl_edge_index_list = []
    rtl_edge_type_list = []
    rtl_edge_labels_list = []
    rtl_batch_list = []
    rtl_node_offset = 0

    # ASM 图
    asm_x_list = []
    asm_edge_index_list = []
    asm_edge_type_list = []
    asm_batch_list = []
    asm_node_offset = 0

    # 图级标签
    graph_label_list = []

    for i, data in enumerate(batch):
        # RTL 节点
        num_rtl_nodes = data.rtl_x.size(0)
        rtl_x_list.append(data.rtl_x)
        rtl_batch_list.append(torch.full((num_rtl_nodes,), i, dtype=torch.long))

        # RTL 边（偏移节点索引）
        rtl_edge_index_list.append(data.rtl_edge_index + rtl_node_offset)
        if data.rtl_edge_type is not None:
            rtl_edge_type_list.append(data.rtl_edge_type)
        if data.rtl_edge_labels is not None:
            rtl_edge_labels_list.append(data.rtl_edge_labels)

        rtl_node_offset += num_rtl_nodes

        # ASM 节点（可能为空）
        if data.asm_x is not None:
            num_asm_nodes = data.asm_x.size(0)
            asm_x_list.append(data.asm_x)
            asm_batch_list.append(torch.full((num_asm_nodes,), i, dtype=torch.long))

            # ASM 边
            if data.asm_edge_index is not None:
                asm_edge_index_list.append(data.asm_edge_index + asm_node_offset)
            if data.asm_edge_type is not None:
                asm_edge_type_list.append(data.asm_edge_type)

            asm_node_offset += num_asm_nodes

        # 图级标签
        if data.graph_label is not None:
            graph_label_list.append(data.graph_label)

    # 拼接 RTL
    rtl_x = torch.cat(rtl_x_list, dim=0)
    rtl_edge_index = torch.cat(rtl_edge_index_list, dim=1)
    rtl_batch = torch.cat(rtl_batch_list, dim=0)
    rtl_edge_type = torch.cat(rtl_edge_type_list, dim=0) if rtl_edge_type_list else None
    rtl_edge_labels = torch.cat(rtl_edge_labels_list, dim=0) if rtl_edge_labels_list else None

    # 拼接 ASM
    if asm_x_list:
        asm_x = torch.cat(asm_x_list, dim=0)
        asm_batch = torch.cat(asm_batch_list, dim=0)
        asm_edge_index = torch.cat(asm_edge_index_list, dim=1) if asm_edge_index_list else None
        asm_edge_type = torch.cat(asm_edge_type_list, dim=0) if asm_edge_type_list else None
    else:
        asm_x = asm_batch = asm_edge_index = asm_edge_type = None

    # 图级标签
    graph_label = torch.cat(graph_label_list, dim=0) if graph_label_list else None

    return DualGraphData(
        rtl_x=rtl_x,
        rtl_edge_index=rtl_edge_index,
        rtl_edge_type=rtl_edge_type,
        rtl_batch=rtl_batch,
        asm_x=asm_x,
        asm_edge_index=asm_edge_index,
        asm_edge_type=asm_edge_type,
        asm_batch=asm_batch,
        rtl_edge_labels=rtl_edge_labels,
        graph_label=graph_label
    )


class DualGraphDataModule(L.LightningDataModule):
    """双图数据加载模块"""

    def __init__(
        self,
        train_data: List[DualGraphData],
        val_data: Optional[List[DualGraphData]] = None,
        test_data: Optional[List[DualGraphData]] = None,
        batch_size: int = 32,
        num_workers: int = 4
    ):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """初始化数据集"""
        if stage == "fit" or stage is None:
            self.train_dataset = DualGraphDataset(self.train_data)
            if self.val_data:
                self.val_dataset = DualGraphDataset(self.val_data)

        if stage == "test" or stage is None:
            if self.test_data:
                self.test_dataset = DualGraphDataset(self.test_data)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_dual_graph,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_dual_graph,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_dual_graph,
            pin_memory=True
        )
