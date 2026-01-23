#!/usr/bin/env python3
"""训练模块测试"""

import torch
from models import ModelConfig, DualGraphData
from trainer import (
    DualGraphLightningModule,
    DualGraphDataModule,
    TrainerConfig,
    collate_dual_graph
)
import lightning as L


def create_dummy_data(
    num_samples: int = 10,
    rtl_node_dim: int = 32,
    asm_node_dim: int = 32
) -> list[DualGraphData]:
    """创建测试用虚拟数据"""
    data_list = []

    for _ in range(num_samples):
        n1 = torch.randint(10, 30, (1,)).item()  # RTL 节点数
        n2 = torch.randint(5, 15, (1,)).item()   # ASM 节点数
        e1 = n1 * 2  # RTL 边数
        e2 = n2 * 2  # ASM 边数

        # 生成边标签：-1, 0, 1
        edge_labels = torch.randint(-1, 2, (e1,))

        data = DualGraphData(
            rtl_x=torch.randn(n1, rtl_node_dim),
            rtl_edge_index=torch.randint(0, n1, (2, e1)),
            rtl_edge_type=torch.randint(0, 5, (e1,)),
            rtl_batch=None,  # 单图时为 None
            asm_x=torch.randn(n2, asm_node_dim),
            asm_edge_index=torch.randint(0, n2, (2, e2)),
            asm_edge_type=torch.randint(0, 5, (e2,)),
            asm_batch=None,
            rtl_edge_labels=edge_labels,
            graph_label=torch.rand(1, 1)
        )
        data_list.append(data)

    return data_list


def test_collate_fn():
    """测试 collate 函数"""
    print("=" * 50)
    print("测试 collate_dual_graph")
    print("=" * 50)

    data_list = create_dummy_data(num_samples=4)
    batch = collate_dual_graph(data_list)

    print(f"✓ RTL 节点: {batch.rtl_x.shape}")
    print(f"✓ RTL 边: {batch.rtl_edge_index.shape}")
    print(f"✓ RTL batch: {batch.rtl_batch.shape}, unique: {batch.rtl_batch.unique().tolist()}")
    print(f"✓ ASM 节点: {batch.asm_x.shape}")
    print(f"✓ 图标签: {batch.graph_label.shape}")

    # 验证 batch 索引正确性
    assert batch.rtl_batch.max().item() == 3, "RTL batch 索引应该是 0-3"
    assert batch.asm_batch.max().item() == 3, "ASM batch 索引应该是 0-3"
    print("✓ Batch 索引验证通过")


def test_datamodule():
    """测试 DataModule"""
    print("\n" + "=" * 50)
    print("测试 DualGraphDataModule")
    print("=" * 50)

    train_data = create_dummy_data(num_samples=20)
    val_data = create_dummy_data(num_samples=5)

    datamodule = DualGraphDataModule(
        train_data=train_data,
        val_data=val_data,
        batch_size=4,
        num_workers=0  # 测试时使用 0
    )

    datamodule.setup("fit")

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    print(f"✓ 训练集 batches: {len(train_loader)}")
    print(f"✓ 验证集 batches: {len(val_loader)}")

    # 测试迭代
    batch = next(iter(train_loader))
    print(f"✓ 获取 batch: RTL nodes={batch.rtl_x.shape[0]}, edges={batch.rtl_edge_index.shape[1]}")


def test_lightning_module():
    """测试 LightningModule"""
    print("\n" + "=" * 50)
    print("测试 DualGraphLightningModule")
    print("=" * 50)

    config = ModelConfig(
        rtl_node_dim=32,
        asm_node_dim=32,
        hidden_dim=64,
        num_gnn_layers=2,
        num_heads=2
    )

    module = DualGraphLightningModule(
        model_config=config,
        learning_rate=1e-3,
        warmup_steps=10,
        scheduler_type="cosine"
    )

    params = sum(p.numel() for p in module.parameters())
    print(f"✓ 模型参数量: {params:,}")

    # 测试前向传播
    data_list = create_dummy_data(num_samples=2, rtl_node_dim=32, asm_node_dim=32)
    batch = collate_dual_graph(data_list)

    module.eval()
    with torch.no_grad():
        output = module(batch)
    print(f"✓ 前向传播: edge_logits={output.edge_logits.shape}, graph_pred={output.graph_pred.shape}")

    # 测试训练步骤
    module.train()
    loss = module.training_step(batch, 0)
    print(f"✓ 训练步骤: loss={loss.item():.4f}")


def test_training_loop():
    """测试完整训练循环（1 epoch）"""
    print("\n" + "=" * 50)
    print("测试完整训练循环")
    print("=" * 50)

    # 创建数据
    train_data = create_dummy_data(num_samples=16, rtl_node_dim=32, asm_node_dim=32)
    val_data = create_dummy_data(num_samples=4, rtl_node_dim=32, asm_node_dim=32)

    # 创建模块
    config = ModelConfig(
        rtl_node_dim=32,
        asm_node_dim=32,
        hidden_dim=64,
        num_gnn_layers=2,
        num_heads=2
    )

    module = DualGraphLightningModule(
        model_config=config,
        learning_rate=1e-3,
        warmup_steps=5,
        scheduler_type="cosine"
    )

    # 创建数据模块
    datamodule = DualGraphDataModule(
        train_data=train_data,
        val_data=val_data,
        batch_size=4,
        num_workers=0
    )

    # 创建 Trainer（快速测试模式）
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False
    )

    # 训练
    trainer.fit(module, datamodule)
    print("✓ 训练循环完成（1 epoch）")

    # 验证
    trainer.validate(module, datamodule)
    print("✓ 验证完成")


def test_trainer_config():
    """测试 TrainerConfig"""
    print("\n" + "=" * 50)
    print("测试 TrainerConfig")
    print("=" * 50)

    config = TrainerConfig(
        learning_rate=1e-4,
        batch_size=32,
        max_epochs=100,
        scheduler_type="cosine"
    )

    print(f"✓ learning_rate: {config.learning_rate}")
    print(f"✓ batch_size: {config.batch_size}")
    print(f"✓ max_epochs: {config.max_epochs}")
    print(f"✓ scheduler_type: {config.scheduler_type}")


if __name__ == "__main__":
    test_collate_fn()
    test_datamodule()
    test_lightning_module()
    test_trainer_config()
    test_training_loop()

    print("\n" + "=" * 50)
    print("🎉 所有测试通过!")
    print("=" * 50)
