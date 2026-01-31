#!/usr/bin/env python3
"""
FiLM 注入模块

实现双向特征注入：
- ASM → RTL：测试激励驱动硬件
- RTL → ASM：硬件状态反馈给激励

基于 FiLM (Feature-wise Linear Modulation) 方法：
- Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer" (AAAI 2018)
"""

import torch
import torch.nn as nn


class FiLMInjection(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) 注入模块

    将上下文特征注入到目标节点特征：
    h' = (1 + γ) ⊙ h + β

    其中 γ, β 由上下文生成。

    设计特点：
    - 使用 (1 + γ) 而非 γ，确保初始时接近恒等映射
    - γ 通过 Tanh 限制范围在 [-1, 1]，稳定训练
    - 小随机初始化确保梯度非零，同时保持初始时接近恒等映射
    """

    def __init__(self, hidden_dim: int, init_std: float = 0.02):
        """
        Args:
            hidden_dim: 隐藏层维度
            init_std: 初始化标准差（默认 0.02，足够小以接近恒等映射）
        """
        super().__init__()

        # γ 生成器（缩放因子）
        self.gamma_gen = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),  # 限制范围在 [-1, 1]，稳定训练
        )

        # β 生成器（偏移量）
        self.beta_gen = nn.Linear(hidden_dim, hidden_dim)

        # 小随机初始化：确保梯度非零，同时 γ ≈ 0, β ≈ 0
        nn.init.normal_(self.gamma_gen[0].weight, std=init_std)
        nn.init.zeros_(self.gamma_gen[0].bias)
        nn.init.normal_(self.beta_gen.weight, std=init_std)
        nn.init.zeros_(self.beta_gen.bias)

    def forward(
        self,
        target_h: torch.Tensor,
        context: torch.Tensor,
        target_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        FiLM 调制

        Args:
            target_h: [N, D] 目标节点特征
            context: [B, D] 上下文向量（global_mean_pool 后）
            target_batch: [N] 目标节点的 batch 索引

        Returns:
            [N, D] 调制后的目标节点特征
        """
        # 生成调制参数
        gamma = self.gamma_gen(context)  # [B, D]
        beta = self.beta_gen(context)  # [B, D]

        # 广播到每个目标节点
        gamma_expanded = gamma[target_batch]  # [N, D]
        beta_expanded = beta[target_batch]  # [N, D]

        # FiLM 调制: h' = (1 + γ) ⊙ h + β
        # 使用 (1 + γ) 确保初始时接近恒等映射
        return (1 + gamma_expanded) * target_h + beta_expanded
