#!/usr/bin/env python3
"""
跨图交互模块 (简化版)

基于论文最佳实践优化：
- Pre-LN 结构 (Xiong et al., 2020)
- 简化门控 (GRU-style)
- 共享 K/V 投影减少参数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CrossGraphAttention(nn.Module):
    """
    简化的跨图注意力

    优化点：
    1. 共享 K/V 投影（RTL 和 ASM 使用同一套）
    2. 简化门控：单层线性 + sigmoid
    3. Pre-LN 结构
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        # 共享的 Q/K/V 投影
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # 简化门控：学习标量缩放
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, 1), nn.Sigmoid())

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [B, N_q, D]
            key_value: [B, N_k, D]
            kv_mask: [B, N_k] True=有效

        Returns:
            [B, N_q, D] 更新后的 query
        """
        B, N_q, D = query.shape
        N_k = key_value.shape[1]

        # Pre-LN
        q_normed = self.norm(query)

        # QKV 投影
        Q = (
            self.q_proj(q_normed)
            .view(B, N_q, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.k_proj(key_value)
            .view(B, N_k, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.v_proj(key_value)
            .view(B, N_k, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # 注意力
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if kv_mask is not None:
            attn = attn.masked_fill(~kv_mask[:, None, None, :], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 聚合
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N_q, D)
        out = self.out_proj(out)

        # 门控残差
        gate_val = self.gate(torch.cat([query, out], dim=-1))  # [B, N_q, 1]
        return query + gate_val * self.dropout(out)


class CrossGraphInteraction(nn.Module):
    """
    双向跨图交互

    RTL ← ASM 和 ASM ← RTL 同时进行
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # 双向注意力（共享参数以减少过拟合）
        self.attn = CrossGraphAttention(hidden_dim, num_heads, dropout)

    def forward(
        self,
        rtl: torch.Tensor,
        asm: torch.Tensor,
        rtl_mask: Optional[torch.Tensor] = None,
        asm_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rtl: [B, N1, D]
            asm: [B, N2, D]

        Returns:
            rtl_updated, asm_updated
        """
        # RTL 从 ASM 聚合信息
        rtl_updated = self.attn(rtl, asm, asm_mask)
        # ASM 从 RTL 聚合信息
        asm_updated = self.attn(asm, rtl, rtl_mask)

        return rtl_updated, asm_updated
