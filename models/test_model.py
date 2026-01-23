#!/usr/bin/env python3
"""模型测试"""

import torch
from models import create_small_model, DualGraphData


def test():
    print("=" * 50)
    print("测试双图融合模型（带边类型）")
    print("=" * 50)

    model = create_small_model(rtl_node_dim=32, asm_node_dim=32)
    params = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型参数量: {params:,}")

    # 测试数据（包含边类型）
    B, N1, N2, E1, E2 = 2, 20, 10, 40, 20

    # 生成边标签：-1（不适用）、0（未覆盖）、1（已覆盖）
    # 约 1/3 为 -1，2/3 为有效标签
    edge_labels = torch.randint(-1, 2, (E1,))

    data = DualGraphData(
        rtl_x=torch.randn(B * N1, 32),
        rtl_edge_index=torch.randint(0, N1, (2, E1)),
        rtl_edge_type=torch.randint(0, 5, (E1,)),  # 5 种边类型
        rtl_batch=torch.repeat_interleave(torch.arange(B), N1),
        asm_x=torch.randn(B * N2, 32),
        asm_edge_index=torch.randint(0, N2, (2, E2)),
        asm_edge_type=torch.randint(0, 5, (E2,)),  # 5 种边类型
        asm_batch=torch.repeat_interleave(torch.arange(B), N2),
        rtl_edge_labels=edge_labels,
        graph_label=torch.rand(B, 1)
    )

    # 前向传播
    model.eval()
    with torch.no_grad():
        out = model(data)
    print(f"✓ 前向传播: edge_logits={out.edge_logits.shape}, graph_pred={out.graph_pred.shape}")

    # 反向传播
    model.train()
    out = model(data)
    loss = model.compute_loss(out, data)
    loss['total_loss'].backward()

    num_valid = loss['num_valid_edges']
    num_masked = E1 - num_valid
    print(f"✓ 边标签统计: 有效边={num_valid}, mask边={num_masked}")
    print(f"✓ 反向传播: loss={loss['total_loss'].item():.4f}")

    # 检查梯度
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"✓ 梯度: {has_grad}/{len(list(model.parameters()))} 参数有梯度")

    # 测试无边类型的情况（向后兼容）
    print("\n" + "-" * 50)
    print("测试无边类型（向后兼容）")
    print("-" * 50)

    data_no_edge_type = DualGraphData(
        rtl_x=torch.randn(B * N1, 32),
        rtl_edge_index=torch.randint(0, N1, (2, E1)),
        rtl_edge_type=None,  # 无边类型
        rtl_batch=torch.repeat_interleave(torch.arange(B), N1),
        asm_x=torch.randn(B * N2, 32),
        asm_edge_index=torch.randint(0, N2, (2, E2)),
        asm_edge_type=None,  # 无边类型
        asm_batch=torch.repeat_interleave(torch.arange(B), N2),
        rtl_edge_labels=torch.randint(0, 2, (E1,)),  # 仅有效标签 {0, 1}
        graph_label=torch.rand(B, 1)
    )

    model.eval()
    with torch.no_grad():
        out = model(data_no_edge_type)
    print(f"✓ 无边类型前向传播: edge_logits={out.edge_logits.shape}")

    # 测试全部 mask 的情况
    print("\n" + "-" * 50)
    print("测试全部边被 mask 的情况")
    print("-" * 50)

    data_all_masked = DualGraphData(
        rtl_x=torch.randn(B * N1, 32),
        rtl_edge_index=torch.randint(0, N1, (2, E1)),
        rtl_edge_type=torch.randint(0, 5, (E1,)),
        rtl_batch=torch.repeat_interleave(torch.arange(B), N1),
        asm_x=torch.randn(B * N2, 32),
        asm_edge_index=torch.randint(0, N2, (2, E2)),
        asm_edge_type=torch.randint(0, 5, (E2,)),
        asm_batch=torch.repeat_interleave(torch.arange(B), N2),
        rtl_edge_labels=torch.full((E1,), -1),  # 全部为 -1
        graph_label=torch.rand(B, 1)
    )

    model.train()
    out = model(data_all_masked)
    loss = model.compute_loss(out, data_all_masked)
    print(f"✓ 全 mask 情况: edge_loss={loss['edge_loss'].item():.4f}, num_valid={loss['num_valid_edges']}")

    print("\n🎉 所有测试通过!")


if __name__ == "__main__":
    test()
