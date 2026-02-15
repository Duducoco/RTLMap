#!/usr/bin/env python3
"""
特征融合模块

实现双向特征注入：
- ASM → RTL：测试激励驱动硬件
- RTL → ASM：硬件状态反馈给激励

支持两种融合方式：
1. FiLM (Feature-wise Linear Modulation)
   - Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer" (AAAI 2018)

2. SSM-FiLM (State Space Model + FiLM)
   - 结合 Mamba 的选择性状态空间机制和 FiLM 的条件注入
   - 用 SSM 处理源图节点序列，生成动态上下文
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Literal
from torch_geometric.nn import global_mean_pool


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


class SimpleSSM(nn.Module):
    """
    简化版选择性状态空间模型 (Selective State Space Model)

    基于 Mamba 的核心思想，但使用纯 PyTorch 实现（无需 CUDA 内核）。

    状态空间方程：
        h_t = Ā · h_{t-1} + B̄ · x_t    # 状态更新
        y_t = C · h_t                   # 输出

    选择性机制：Δ, B, C 是输入依赖的，模型学习"哪些输入重要"。

    参考：
    - Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2024)
    """

    def __init__(self, d_model: int, d_state: int = 16):
        """
        Args:
            d_model: 模型维度（输入/输出维度）
            d_state: 状态空间维度（隐状态大小）
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # 状态空间参数 A（可学习，初始化为负值以确保稳定性）
        # A 控制状态衰减，负值确保长期稳定
        self.A_log = nn.Parameter(torch.log(torch.rand(d_model, d_state) * 0.5 + 0.5))

        # 选择性参数生成器（输入依赖）
        self.delta_proj = nn.Linear(d_model, d_model)  # Δ: 步长/记忆门
        self.B_proj = nn.Linear(d_model, d_state)  # B: 输入门
        self.C_proj = nn.Linear(d_model, d_state)  # C: 输出门

        # 输出投影（可选，用于残差连接）
        self.out_proj = nn.Linear(d_model, d_model)

        # 初始化
        nn.init.xavier_uniform_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)
        nn.init.xavier_uniform_(self.B_proj.weight)
        nn.init.zeros_(self.B_proj.bias)
        nn.init.xavier_uniform_(self.C_proj.weight)
        nn.init.zeros_(self.C_proj.bias)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        前向传播

        Args:
            x: [B, T, D] 输入序列
            mask: [B, T] 可选的 padding mask（True 表示有效位置）

        Returns:
            y: [B, T, D] 输出序列
        """
        B, T, D = x.shape

        # 获取 A（负值确保稳定性）
        A = -torch.exp(self.A_log)  # [D, d_state]

        # 选择性参数（输入依赖）
        delta = F.softplus(self.delta_proj(x))  # [B, T, D] 步长，正值
        B_t = self.B_proj(x)  # [B, T, d_state] 输入门
        C_t = self.C_proj(x)  # [B, T, d_state] 输出门

        # 离散化：Ā = exp(Δ · A)
        # delta: [B, T, D], A: [D, d_state] -> delta_A: [B, T, D, d_state]
        delta_A = delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(
            0
        )  # [B, T, D, d_state]
        A_bar = torch.exp(delta_A)  # [B, T, D, d_state]

        # 状态递推
        h = x.new_zeros(B, D, self.d_state)  # [B, D, d_state] 初始状态
        outputs = []

        for t in range(T):
            # 状态更新: h_t = Ā · h_{t-1} + Δ · B · x_t
            # A_bar[:, t]: [B, D, d_state]
            # delta[:, t]: [B, D]
            # B_t[:, t]: [B, d_state]
            # x[:, t]: [B, D]
            delta_B_x = delta[:, t, :, None] * B_t[:, t, None, :] * x[:, t, :, None]
            # delta_B_x: [B, D, d_state]
            h = A_bar[:, t] * h + delta_B_x

            # 输出: y_t = C · h_t
            # h: [B, D, d_state], C_t[:, t]: [B, d_state]
            y_t = (h * C_t[:, t, None, :]).sum(-1)  # [B, D]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # [B, T, D]

        # 输出投影
        y = self.out_proj(y)

        return y


class SSMFiLMInjection(nn.Module):
    """
    SSM 增强的 FiLM 注入模块

    用 SSM 处理源图节点序列，生成动态上下文，然后进行 FiLM 注入。

    相比原始 FiLM 的优势：
    - 动态上下文：SSM 状态递推模拟"执行状态演化"
    - 选择性记忆：学习"哪些源节点对目标重要"
    - 长程依赖：SSM 天然支持长序列

    架构：
    1. 将源节点按 batch 分组并 padding 成序列
    2. SSM 处理序列，生成每个位置的状态
    3. 选择性聚合状态生成上下文
    4. FiLM 注入到目标节点
    """

    def __init__(
        self,
        hidden_dim: int,
        d_state: int = 16,
        pool_mode: Literal["last", "mean", "attention"] = "last",
        init_std: float = 0.02,
    ):
        """
        Args:
            hidden_dim: 隐藏层维度
            d_state: SSM 状态空间维度
            pool_mode: 聚合模式
                - "last": 取最后一个有效状态
                - "mean": 平均池化
                - "attention": 注意力聚合
            init_std: FiLM 参数初始化标准差
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pool_mode = pool_mode

        # SSM 模块
        self.ssm = SimpleSSM(hidden_dim, d_state)

        # 注意力聚合（可选）
        if pool_mode == "attention":
            self.query_proj = nn.Linear(hidden_dim, hidden_dim)
            self.key_proj = nn.Linear(hidden_dim, hidden_dim)

        # FiLM 参数生成器
        self.gamma_gen = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.beta_gen = nn.Linear(hidden_dim, hidden_dim)

        # 小随机初始化
        nn.init.normal_(self.gamma_gen[0].weight, std=init_std)
        nn.init.zeros_(self.gamma_gen[0].bias)
        nn.init.normal_(self.beta_gen.weight, std=init_std)
        nn.init.zeros_(self.beta_gen.bias)

    def forward(
        self,
        target_h: Tensor,
        source_h: Tensor,
        target_batch: Tensor,
        source_batch: Tensor,
    ) -> Tensor:
        """
        SSM-FiLM 调制

        Args:
            target_h: [N_tgt, D] 目标节点特征（如 RTL）
            source_h: [N_src, D] 源节点特征（如 ASM）
            target_batch: [N_tgt] 目标节点的 batch 索引
            source_batch: [N_src] 源节点的 batch 索引

        Returns:
            [N_tgt, D] 调制后的目标节点特征
        """
        # 1. 将源节点按 batch 分组并 padding 成序列
        source_seq, seq_mask = self._batch_to_sequence(source_h, source_batch)
        # source_seq: [B, T_max, D], seq_mask: [B, T_max]

        # 2. SSM 处理
        ssm_out = self.ssm(source_seq, seq_mask)  # [B, T_max, D]

        # 3. 选择性聚合
        context = self._aggregate(ssm_out, seq_mask, target_h, target_batch)
        # context: [B, D]

        # 4. FiLM 注入
        gamma = self.gamma_gen(context)  # [B, D]
        beta = self.beta_gen(context)  # [B, D]

        gamma_expanded = gamma[target_batch]  # [N_tgt, D]
        beta_expanded = beta[target_batch]  # [N_tgt, D]

        return (1 + gamma_expanded) * target_h + beta_expanded

    def _batch_to_sequence(self, h: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        """
        将 batch 格式转换为 padding 序列格式

        Args:
            h: [N, D] 节点特征
            batch: [N] batch 索引

        Returns:
            seq: [B, T_max, D] padding 后的序列
            mask: [B, T_max] 有效位置 mask（True 表示有效）
        """
        device = h.device
        batch_size = batch.max().item() + 1

        # 计算每个 batch 的节点数
        counts = torch.bincount(batch, minlength=batch_size)
        max_len = counts.max().item()

        # 初始化
        seq = h.new_zeros(batch_size, max_len, h.size(-1))
        mask = h.new_zeros(batch_size, max_len, dtype=torch.bool)

        # 填充（使用向量化操作优化）
        for b in range(batch_size):
            idx = (batch == b).nonzero(as_tuple=True)[0]
            length = len(idx)
            seq[b, :length] = h[idx]
            mask[b, :length] = True

        return seq, mask

    def _aggregate(
        self,
        ssm_out: Tensor,
        seq_mask: Tensor,
        target_h: Tensor,
        target_batch: Tensor,
    ) -> Tensor:
        """
        选择性聚合 SSM 输出

        Args:
            ssm_out: [B, T, D] SSM 输出
            seq_mask: [B, T] 有效位置 mask
            target_h: [N_tgt, D] 目标节点特征
            target_batch: [N_tgt] 目标节点 batch 索引

        Returns:
            context: [B, D] 聚合后的上下文
        """
        B, T, D = ssm_out.shape

        if self.pool_mode == "last":
            # 取每个序列的最后一个有效位置
            lengths = seq_mask.sum(dim=1).long()  # [B]
            # 处理空序列的情况
            lengths = lengths.clamp(min=1) - 1
            context = ssm_out[torch.arange(B, device=ssm_out.device), lengths]  # [B, D]

        elif self.pool_mode == "mean":
            # 平均池化（mask 掉 padding）
            ssm_masked = ssm_out * seq_mask.unsqueeze(-1).float()
            lengths = seq_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
            context = ssm_masked.sum(dim=1) / lengths  # [B, D]

        elif self.pool_mode == "attention":
            # 注意力聚合：目标图级特征作为 query
            target_graph = global_mean_pool(target_h, target_batch)  # [B, D]
            query = self.query_proj(target_graph).unsqueeze(1)  # [B, 1, D]
            key = self.key_proj(ssm_out)  # [B, T, D]

            # 计算注意力分数
            attn = torch.bmm(query, key.transpose(1, 2)) / (self.hidden_dim**0.5)
            # [B, 1, T]

            # mask 掉 padding 位置
            attn = attn.masked_fill(~seq_mask.unsqueeze(1), float("-inf"))
            attn = F.softmax(attn, dim=-1)  # [B, 1, T]

            # 加权聚合
            context = torch.bmm(attn, ssm_out).squeeze(1)  # [B, D]

        else:
            raise ValueError(f"Unknown pool_mode: {self.pool_mode}")

        return context
