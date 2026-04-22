#!/usr/bin/env python3
"""
特征融合模块：PerceiverCrossFusion

用 K 个可学习 latent token 作为跨图信息瓶颈，实现双向节点级融合：
  ASM → RTL：测试激励驱动硬件
  RTL → ASM：硬件状态反馈给激励

复杂度 O((N_src + N_tgt) · K · D)，消除 N_rtl × N_asm 二次项。
数值稳定：pre/post LayerNorm、key_padding_mask、gate bias=-3、mlp zero-init。
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import to_dense_batch


class PerceiverCrossFusion(nn.Module):
    """
    Perceiver-IO 风格的跨图双向融合模块

    流程（每方向一个实例）：
      1. attn_pool：latent ← source（K 个 latent token 汇聚源图信息）
      2. attn_read：target ← latent（将摘要注入目标节点，节点级 cross-attn）
      3. Gated 残差：Highway gate 控制注入强度

    数值稳定性保证：
      - pre-LayerNorm（源侧 / 目标侧）+ post-LayerNorm（context）
      - key_padding_mask 走 PyTorch 内置 stable softmax，不手动 fill -inf
      - gate bias=-3 → sigmoid≈0.05，起步近恒等映射
      - mlp 最后线性层 zero-init，起步输出为零残差
      - 全空 batch 行（无有效源节点）：latents 置零，不污染其他样本
    """

    def __init__(
        self,
        hidden_dim: int,
        num_latents: int = 16,
        num_heads: int = 4,
        num_layers: int = 4,
        init_std: float = 0.02,
    ):
        """
        Args:
            hidden_dim:  模型隐藏维度 D
            num_latents: latent token 数量 K（建议 8–16）
            num_heads:   Multi-head Attention 头数
            num_layers:  总 GNN 层数（用于 mlp 输出投影 GPT-NeoX 风格缩放）
            init_std:    gate/latent 初始化标准差
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # 可学习 latent tokens [K, D]
        self.latents = nn.Parameter(torch.empty(num_latents, hidden_dim))
        nn.init.trunc_normal_(self.latents, std=init_std)

        # pre-LayerNorm
        self.ln_src = nn.LayerNorm(hidden_dim)
        self.ln_tgt = nn.LayerNorm(hidden_dim)
        # post-LayerNorm（context 向量）
        self.ln_ctx = nn.LayerNorm(hidden_dim)

        # latent ← source 池化（Q=latents, KV=source）
        self.attn_pool = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0,
        )

        # target ← latent 读取（Q=target, KV=latents）
        self.attn_read = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0,
        )

        # Highway gate：σ(W[h; ctx])，bias=-3 → 起步 σ≈0.05
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.normal_(self.gate.weight, std=init_std)
        nn.init.constant_(self.gate.bias, -3.0)

        # 注入 MLP（最后层 zero-init）
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        scale = 1.0 / math.sqrt(2.0 * max(num_layers, 1))
        nn.init.normal_(self.mlp[-1].weight, std=scale * init_std)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        target_h: Tensor,
        source_h: Tensor,
        target_batch: Tensor,
        source_batch: Tensor,
    ) -> Tensor:
        """
        Args:
            target_h:     [N_tgt, D] 目标图节点特征（如 RTL）
            source_h:     [N_src, D] 源图节点特征（如 ASM）
            target_batch: [N_tgt]    目标节点 batch 索引
            source_batch: [N_src]    源节点 batch 索引
        Returns:
            [N_tgt, D] 融合后目标节点特征
        """
        # ── Step 1：源图 → latents ──────────────────────────────────────────
        src_seq, src_mask = to_dense_batch(source_h, source_batch)  # [B,T_s,D], [B,T_s]
        B = src_seq.size(0)

        q = self.latents.unsqueeze(0).expand(B, -1, -1)  # [B,K,D]
        # key_padding_mask: True = 忽略（PyTorch MHA 约定）
        src_normed = self.ln_src(src_seq)
        Z, _ = self.attn_pool(
            q,
            src_normed,
            src_normed,
            key_padding_mask=~src_mask,
        )  # [B,K,D]

        # 全空行（无有效源节点）：latent 置零，不污染其他样本
        has_src = src_mask.any(dim=1)  # [B]
        Z = torch.where(has_src[:, None, None], Z, Z.new_zeros(()))

        # ── Step 2：target ← latents ────────────────────────────────────────
        tgt_seq, tgt_mask = to_dense_batch(target_h, target_batch)  # [B,T_t,D], [B,T_t]

        ctx_seq, _ = self.attn_read(
            self.ln_tgt(tgt_seq),
            Z,
            Z,
        )  # [B,T_t,D]

        # dense → flat（只取有效位置）
        ctx = ctx_seq[tgt_mask]  # [N_tgt, D]

        # ── Step 3：Gated 残差注入 ───────────────────────────────────────────
        ctx_normed = self.ln_ctx(ctx)
        g = torch.sigmoid(self.gate(torch.cat([target_h, ctx_normed], dim=-1)))  # [N_tgt,D]
        return target_h + g * self.mlp(ctx_normed)
