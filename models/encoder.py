#!/usr/bin/env python3
"""
图编码器模块

RTL 节点编码：node_cell_type + node_width
RTL 边编码：edge_type + edge_width

ASM 节点编码：node_type + instruction_encoding（外部模型生成）
ASM 边编码：edge_type（仅类型嵌入）

设计选择：
- 使用 MessagePassing 实现支持边特征更新的 GNN 层
- RTL 和 ASM 使用独立的 GNN 层和边类型嵌入
- DropPath 替代 Dropout（更稳定）
- 双向 FiLM 注入实现 ASM ↔ RTL 双向特征调制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing, global_mean_pool
from typing import Optional, Tuple

from .interaction import FiLMInjection, SSMFiLMInjection

# EdgeType 索引常量（0-indexed，用于控制门控聚合和 Embedding 层）
# 对应 cdfg_rtl.data_types.EdgeType 枚举（值需减 1 转为 0-indexed）
EDGE_TYPE_DATA = 0  # 通用数据流
EDGE_TYPE_DATA_TRUE = 1  # MUX 真分支数据流（B 端口，S=1 时选择）
EDGE_TYPE_DATA_FALSE = 2  # MUX 假分支数据流（A 端口，S=0 时选择）
EDGE_TYPE_CONTROL = 3  # 控制流（MUX S 端口）
EDGE_TYPE_CLOCK = 4  # 时钟
EDGE_TYPE_RESET = 5  # 复位
EDGE_TYPE_ENABLE = 6  # 使能


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Stochastic Depth (论文: Deep Networks with Stochastic Depth)"""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()
    return x.div(keep_prob) * mask


class RTLNodeFeatureEncoder(nn.Module):
    """
    RTL 节点特征编码器

    将 cell_type 索引和 width 编码为节点嵌入：
    - cell_type: 通过 Embedding 层嵌入
    - width: 使用 log2(width + 1) 编码后通过线性层
    - 融合方式: 加法融合
    """

    def __init__(self, num_types: int, hidden_dim: int):
        super().__init__()
        self.type_embedding = nn.Embedding(num_types, hidden_dim)
        self.width_encoder = nn.Linear(1, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, node_type: torch.Tensor, node_width: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            node_type: [N] 节点类型索引
            node_width: [N] 节点 width

        Returns:
            [N, hidden_dim] 节点嵌入
        """
        type_emb = self.type_embedding(node_type)  # [N, D]
        width_log = torch.log2(node_width.float() + 1).unsqueeze(-1)  # [N, 1]
        width_emb = self.width_encoder(width_log)  # [N, D]
        return self.norm(type_emb + width_emb)


class AsmNodeFeatureEncoder(nn.Module):
    """
    ASM 节点特征编码器

    将 node_type 和 instruction_encoding 编码为节点嵌入：
    - node_type: 通过 Embedding 层嵌入
    - instruction_encoding: 通过线性层投影（由外部模型生成）
    - 融合方式: 加法融合
    """

    def __init__(self, num_node_types: int, instruction_dim: int, hidden_dim: int):
        """
        Args:
            num_node_types: ASM 节点类型数量 (AsmNodeType 枚举)
            instruction_dim: 指令编码维度（由外部模型生成）
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.node_type_embedding = nn.Embedding(num_node_types, hidden_dim)
        self.instruction_proj = nn.Linear(instruction_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, node_type: torch.Tensor, instruction_encoding: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            node_type: [M] ASM 节点类型索引
            instruction_encoding: [M, D_instr] 指令编码（由外部模型生成）

        Returns:
            [M, hidden_dim] 节点嵌入
        """
        type_emb = self.node_type_embedding(node_type)  # [M, D]
        instr_emb = self.instruction_proj(instruction_encoding)  # [M, D]
        return self.norm(type_emb + instr_emb)


class RTLEdgeFeatureEncoder(nn.Module):
    """
    RTL 边特征编码器

    将 edge_type、width 和端口位置索引编码为边嵌入：
    - edge_type: 通过 Embedding 层嵌入
    - width: 使用 log2(width + 1) 编码后通过线性层
    - source_port_idx: 源端口在源节点 output_ports 中的位置索引
    - target_port_idx: 目标端口在目标节点 input_ports 中的位置索引
    - 融合方式: 加法融合
    """

    def __init__(self, num_edge_types: int, hidden_dim: int, max_ports: int = 8):
        super().__init__()
        self.edge_type_embedding = nn.Embedding(num_edge_types, hidden_dim)
        self.width_encoder = nn.Linear(1, hidden_dim)
        # 端口位置 embedding：各 hidden_dim // 2，连接后为 hidden_dim
        self.source_port_embedding = nn.Embedding(max_ports, hidden_dim // 2)
        self.target_port_embedding = nn.Embedding(max_ports, hidden_dim // 2)

    def forward(
        self,
        edge_type: torch.Tensor,
        edge_width: Optional[torch.Tensor],
        source_port_idx: Optional[torch.Tensor] = None,
        target_port_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            edge_type: [E] 边类型索引
            edge_width: [E] 边 width（可选）
            source_port_idx: [E] 源端口位置索引（可选）
            target_port_idx: [E] 目标端口位置索引（可选）

        Returns:
            [E, hidden_dim] 边嵌入
        """
        type_emb = self.edge_type_embedding(edge_type)  # [E, D]

        result = type_emb

        if edge_width is not None:
            width_log = torch.log2(edge_width.float() + 1).unsqueeze(-1)  # [E, 1]
            width_emb = self.width_encoder(width_log)  # [E, D]
            result = result + width_emb

        if source_port_idx is not None and target_port_idx is not None:
            # 两个端口索引都存在：连接融合
            source_port_emb = self.source_port_embedding(source_port_idx)  # [E, D/2]
            target_port_emb = self.target_port_embedding(target_port_idx)  # [E, D/2]
            port_emb = torch.cat([source_port_emb, target_port_emb], dim=-1)  # [E, D]
            result = result + port_emb
        elif source_port_idx is not None:
            # 仅有 source_port_idx：用零填充 target 部分
            source_port_emb = self.source_port_embedding(source_port_idx)  # [E, D/2]
            port_emb = torch.cat(
                [source_port_emb, torch.zeros_like(source_port_emb)], dim=-1
            )
            result = result + port_emb
        elif target_port_idx is not None:
            # 仅有 target_port_idx：用零填充 source 部分
            target_port_emb = self.target_port_embedding(target_port_idx)  # [E, D/2]
            port_emb = torch.cat(
                [torch.zeros_like(target_port_emb), target_port_emb], dim=-1
            )
            result = result + port_emb

        return result


class AsmEdgeFeatureEncoder(nn.Module):
    """
    ASM 边特征编码器

    仅使用 edge_type 嵌入（ASM 边不需要 width）
    """

    def __init__(self, num_edge_types: int, hidden_dim: int):
        super().__init__()
        self.edge_type_embedding = nn.Embedding(num_edge_types, hidden_dim)

    def forward(self, edge_type: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_type: [F] 边类型索引

        Returns:
            [F, hidden_dim] 边嵌入
        """
        return self.edge_type_embedding(edge_type)


class GNNLayer(MessagePassing):
    """
    支持边特征更新的 GNN 层

    每次前向传播同时更新：
    1. 节点特征：通过消息聚合（邻居特征 + 边特征）
    2. 边特征：通过源/目标节点特征和当前边特征

    使用残差连接和 LayerNorm 确保训练稳定性
    """

    def __init__(self, hidden_dim: int, drop_path_rate: float = 0.1):
        super().__init__(aggr="add")
        self.hidden_dim = hidden_dim
        self.drop_path_rate = drop_path_rate

        # 节点更新 MLP: [neighbor, edge] -> node_update
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # 边更新 MLP: [src, tgt, edge] -> edge_update
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [N, D] 节点特征
            edge_index: [2, E] 边索引
            edge_attr: [E, D] 边特征

        Returns:
            x: [N, D] 更新后的节点特征
            edge_attr: [E, D] 更新后的边特征
        """
        # 1. 更新节点特征
        node_out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        node_out = drop_path(node_out, self.drop_path_rate, self.training)
        x = self.node_norm(x + node_out)

        # 2. 更新边特征
        edge_out = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)
        edge_out = drop_path(edge_out, self.drop_path_rate, self.training)
        edge_attr = self.edge_norm(edge_attr + edge_out)

        return x, edge_attr

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """消息函数：邻居特征 + 边特征"""
        return self.node_mlp(torch.cat([x_j, edge_attr], dim=-1))

    def edge_update(
        self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """边更新函数：源节点 + 目标节点 + 当前边"""
        return self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


class ControlGatedGNNLayer(MessagePassing):
    """
    控制门控聚合的 GNN 层，精确模拟 MUX 的互斥选择行为

    使用 PyG MessagePassing 框架：
    - propagate() 触发消息传递流程
    - message() 计算边消息（控制边 vs 数据边分别处理）
    - aggregate() 实现控制门控聚合（MUX 节点）或标准聚合（非 MUX 节点）
    - edge_updater() + edge_update() 更新边特征（仅依赖源节点）

    对于 MUX 节点（有 CONTROL 边）：
    - 控制边生成门控权重 [α_true, α_false]
    - 门控权重决定 data_true 和 data_false 的贡献比例
    - softmax 确保互斥选择（权重和为 1）

    对于非 MUX 节点（无 CONTROL 边）：
    - 回退到标准 sum 聚合

    边更新仅依赖源节点（符合 RTL 电路的物理语义）
    """

    def __init__(self, hidden_dim: int, drop_path_rate: float = 0.1):
        # aggr=None: 使用自定义 aggregate()
        # flow='source_to_target': 消息从源节点流向目标节点
        super().__init__(aggr=None, flow="source_to_target")
        self.hidden_dim = hidden_dim
        self.drop_path_rate = drop_path_rate

        # φ: 数据边消息函数（DATA, DATA_TRUE, DATA_FALSE, CLOCK, RESET, ENABLE）
        self.data_message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # φ_ctrl: 控制边消息函数
        self.ctrl_message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # f_gate: 门控函数（控制消息 → 选择权重）
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),  # [gate_true, gate_false]
        )

        # γ: 节点更新函数
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # ψ: 边更新函数（仅依赖源节点，不依赖目标节点）
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_type: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        前向传播

        Args:
            x: [N, D] 节点特征
            edge_index: [2, E] 边索引
            edge_attr: [E, D] 边特征（已通过 RTLEdgeFeatureEncoder 编码）
            edge_type: [E] EdgeType 索引（用于控制门控聚合）

        Returns:
            x: [N, D] 更新后的节点特征
            edge_attr: [E, D] 更新后的边特征
        """
        # 1. 消息传递 + 聚合（通过 propagate 调用 message 和 aggregate）
        node_out = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
            edge_type=edge_type,
            size=None,
        )

        # 2. 节点更新 γ(h_i, m_i)（残差 + DropPath + LayerNorm）
        node_update = self.node_mlp(torch.cat([x, node_out], dim=-1))
        node_update = drop_path(node_update, self.drop_path_rate, self.training)
        x = self.node_norm(x + node_update)

        # 3. 边更新 ψ(h_j, e_{j→i})（通过 edge_updater 调用 edge_update）
        edge_out = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)
        edge_out = drop_path(edge_out, self.drop_path_rate, self.training)
        edge_attr = self.edge_norm(edge_attr + edge_out)

        return x, edge_attr

    def message(
        self,
        x_j: Tensor,
        edge_attr: Tensor,
        edge_type: Tensor,
    ) -> Tensor:
        """
        计算边消息 φ(h_j, e_{j→i})

        PyG 自动将 x 映射为 x_j（源节点特征，因为 flow='source_to_target'）

        Args:
            x_j: [E, D] 源节点特征（PyG 自动索引）
            edge_attr: [E, D] 边特征
            edge_type: [E] 边类型索引

        Returns:
            messages: [E, D] 边消息
        """
        is_ctrl = edge_type == EDGE_TYPE_CONTROL
        is_data = ~is_ctrl

        # 无条件计算所有边消息，用 torch.where 按类型选择（避免 .any() GPU-CPU 同步）
        ctrl_input = torch.cat([x_j, edge_attr], dim=-1)
        ctrl_msg = self.ctrl_message_mlp(ctrl_input).to(x_j.dtype)
        data_msg = self.data_message_mlp(ctrl_input).to(x_j.dtype)

        return torch.where(is_ctrl.unsqueeze(-1), ctrl_msg, data_msg)

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        edge_type: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        """
        控制门控聚合

        MUX 节点: m_gated = α_true · m_true + α_false · m_false + m_other
        非 MUX 节点: m = Σ(all_messages)

        Args:
            inputs: [E, D] 消息（来自 message()）
            index: [E] 目标节点索引（PyG 自动传入）
            edge_type: [E] 边类型索引（从 propagate 传入）
            ptr: 可选，CSR 格式指针
            dim_size: 节点数量

        Returns:
            aggregated: [N, D] 聚合后的消息
        """
        num_nodes = dim_size if dim_size is not None else int(index.max()) + 1
        D = self.hidden_dim

        # 按边类型创建 mask
        is_ctrl = edge_type == EDGE_TYPE_CONTROL
        is_true = edge_type == EDGE_TYPE_DATA_TRUE
        is_false = edge_type == EDGE_TYPE_DATA_FALSE
        is_other = ~(is_ctrl | is_true | is_false)

        # 初始化各类消息聚合结果
        agg_ctrl = inputs.new_zeros(num_nodes, D)
        agg_true = inputs.new_zeros(num_nodes, D)
        agg_false = inputs.new_zeros(num_nodes, D)
        agg_other = inputs.new_zeros(num_nodes, D)

        def expand_idx(mask: Tensor) -> Tensor:
            return index[mask].unsqueeze(-1).expand(-1, D)

        # 分类聚合消息（scatter_add_ 对空 tensor 是安全 no-op，无需 .any() 保护）
        agg_ctrl.scatter_add_(0, expand_idx(is_ctrl), inputs[is_ctrl])
        agg_true.scatter_add_(0, expand_idx(is_true), inputs[is_true])
        agg_false.scatter_add_(0, expand_idx(is_false), inputs[is_false])
        agg_other.scatter_add_(0, expand_idx(is_other), inputs[is_other])

        # 检测哪些节点有 CONTROL 边（MUX 节点）
        has_ctrl = inputs.new_zeros(num_nodes, dtype=torch.bool)
        has_ctrl.scatter_(0, index[is_ctrl], True)

        # 生成门控权重 [α_true, α_false]
        gate_logits = self.gate_mlp(agg_ctrl)  # [N, 2]
        gates = F.softmax(gate_logits, dim=-1)
        alpha_true = gates[:, 0:1]  # [N, 1]
        alpha_false = gates[:, 1:2]  # [N, 1]

        # MUX 节点：门控加权聚合
        gated_out = alpha_true * agg_true + alpha_false * agg_false + agg_other

        # 非 MUX 节点：标准 sum 聚合（所有消息直接相加）
        standard_out = agg_ctrl + agg_true + agg_false + agg_other

        # 根据节点类型选择输出
        return torch.where(has_ctrl.unsqueeze(-1), gated_out, standard_out)

    def edge_update(
        self,
        x_j: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        """
        边更新 ψ(h_j, e_{j→i})

        仅依赖源节点特征，不依赖目标节点（符合 RTL 电路物理语义）

        Args:
            x_j: [E, D] 源节点特征（PyG 自动索引）
            edge_attr: [E, D] 当前边特征

        Returns:
            edge_out: [E, D] 边特征更新量
        """
        return self.edge_mlp(torch.cat([x_j, edge_attr], dim=-1))


class FiLMDualEncoder(nn.Module):
    """
    双向 FiLM 注入式双图编码器

    架构：每层 GNN 后，ASM 和 RTL 通过双向 FiLM 相互注入

    特性：
    - 双向注入：ASM ↔ RTL（测试激励与硬件相互影响）
    - 计算高效：无需 dense batch 转换，直接广播
    - 训练稳定：FiLM 小随机初始化，初始时接近恒等映射且梯度非零
    - 仅调制节点特征，边特征保持不变
    - 支持两种融合模式：FiLM 和 SSM-FiLM
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 4,
        dropout: float = 0.1,
        # RTL 特征配置
        num_cell_types: int = 74,
        num_edge_types: int = 7,
        max_ports: int = 8,  # 端口位置索引最大值
        # ASM 特征配置
        num_asm_node_types: int = 22,
        num_asm_edge_types: int = 10,
        asm_instruction_dim: int = 256,
        # 融合配置
        fusion_type: str = "film",  # "film" 或 "ssm_film"
        ssm_d_state: int = 16,  # SSM 状态空间维度
        ssm_pool_mode: str = "last",  # SSM 聚合模式: "last", "mean", "attention"
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.fusion_type = fusion_type

        # RTL 节点编码器
        self.rtl_node_encoder = RTLNodeFeatureEncoder(num_cell_types, hidden_dim)

        # ASM 节点编码器
        self.asm_node_encoder = AsmNodeFeatureEncoder(
            num_asm_node_types, asm_instruction_dim, hidden_dim
        )

        # RTL 边特征编码器（支持端口位置索引）
        self.rtl_edge_encoder = RTLEdgeFeatureEncoder(
            num_edge_types, hidden_dim, max_ports
        )

        # ASM 边特征编码器（仅类型嵌入）
        self.asm_edge_encoder = AsmEdgeFeatureEncoder(num_asm_edge_types, hidden_dim)

        # GNN 层（独立，逐层增加 drop_path 概率）
        drop_rates = [
            dropout * i / (num_layers - 1) if num_layers > 1 else 0
            for i in range(num_layers)
        ]
        # RTL 使用 ControlGatedGNNLayer（控制门控聚合，模拟 MUX 互斥选择）
        self.rtl_layers = nn.ModuleList(
            [ControlGatedGNNLayer(hidden_dim, drop_rates[i]) for i in range(num_layers)]
        )
        self.asm_layers = nn.ModuleList(
            [GNNLayer(hidden_dim, drop_rates[i]) for i in range(num_layers)]
        )

        # 双向融合层（根据配置选择 FiLM 或 SSM-FiLM）
        if fusion_type == "film":
            self.film_asm2rtl = nn.ModuleList(
                [FiLMInjection(hidden_dim) for _ in range(num_layers)]
            )
            self.film_rtl2asm = nn.ModuleList(
                [FiLMInjection(hidden_dim) for _ in range(num_layers)]
            )
        elif fusion_type == "ssm_film":
            self.film_asm2rtl = nn.ModuleList(
                [
                    SSMFiLMInjection(hidden_dim, ssm_d_state, ssm_pool_mode)
                    for _ in range(num_layers)
                ]
            )
            self.film_rtl2asm = nn.ModuleList(
                [
                    SSMFiLMInjection(hidden_dim, ssm_d_state, ssm_pool_mode)
                    for _ in range(num_layers)
                ]
            )
        else:
            raise ValueError(
                f"Unknown fusion_type: {fusion_type}. Use 'film' or 'ssm_film'."
            )

        # 输出投影（独立）
        self.rtl_output = nn.Linear(hidden_dim, output_dim)
        self.asm_output = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        # RTL 图
        rtl_edge_index: torch.Tensor,
        rtl_node_cell_type: torch.Tensor,
        rtl_node_width: torch.Tensor,
        rtl_edge_type: torch.Tensor,
        rtl_batch: Optional[torch.Tensor] = None,
        rtl_edge_width: Optional[torch.Tensor] = None,
        rtl_edge_source_port_idx: Optional[torch.Tensor] = None,
        rtl_edge_target_port_idx: Optional[torch.Tensor] = None,
        # ASM 图
        asm_edge_index: torch.Tensor = None,
        asm_node_type: torch.Tensor = None,
        asm_instruction_encoding: torch.Tensor = None,
        asm_edge_type: torch.Tensor = None,
        asm_batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            rtl_edge_index: [2, E1] RTL 边索引
            rtl_node_cell_type: [N1] RTL 节点 cell_type 索引
            rtl_node_width: [N1] RTL 节点 width
            rtl_edge_type: [E1] RTL 边类型
            rtl_batch: [N1] RTL batch 索引
            rtl_edge_width: [E1] RTL 边 width
            rtl_edge_source_port_idx: [E1] RTL 边源端口位置索引
            rtl_edge_target_port_idx: [E1] RTL 边目标端口位置索引
            asm_edge_index: [2, E2] ASM 边索引
            asm_node_type: [N2] ASM 节点类型索引
            asm_instruction_encoding: [N2, D_instr] ASM 指令编码
            asm_edge_type: [E2] ASM 边类型
            asm_batch: [N2] ASM batch 索引

        Returns:
            rtl_node: [N1, D] RTL 节点嵌入
            rtl_edge_attr: [E1, D] RTL 边嵌入
            rtl_graph: [B, D] RTL 图级嵌入
            asm_node: [N2, D] ASM 节点嵌入
            asm_graph: [B, D] ASM 图级嵌入
        """
        # 处理单图情况（无 batch 索引）
        if rtl_batch is None:
            rtl_batch = torch.zeros(
                rtl_node_cell_type.size(0),
                dtype=torch.long,
                device=rtl_node_cell_type.device,
            )
        if asm_batch is None:
            asm_batch = torch.zeros(
                asm_node_type.size(0), dtype=torch.long, device=asm_node_type.device
            )

        # 编码节点特征
        rtl_h = self.rtl_node_encoder(rtl_node_cell_type, rtl_node_width)
        asm_h = self.asm_node_encoder(asm_node_type, asm_instruction_encoding)

        # 边特征编码（包含端口位置索引）
        rtl_edge_attr = self.rtl_edge_encoder(
            rtl_edge_type,
            rtl_edge_width,
            rtl_edge_source_port_idx,
            rtl_edge_target_port_idx,
        )
        asm_edge_attr = self.asm_edge_encoder(asm_edge_type)

        for i in range(self.num_layers):
            # 1. 并行 GNN 传播
            asm_h, asm_edge_attr = self.asm_layers[i](
                asm_h, asm_edge_index, asm_edge_attr
            )
            rtl_h, rtl_edge_attr = self.rtl_layers[i](
                rtl_h, rtl_edge_index, rtl_edge_attr, rtl_edge_type
            )

            # 2. 双向融合注入（仅节点特征）
            if self.fusion_type == "film":
                # FiLM: 使用全局池化的上下文
                asm_context = global_mean_pool(asm_h, asm_batch)  # [B, D]
                rtl_context = global_mean_pool(rtl_h, rtl_batch)  # [B, D]
                rtl_h = self.film_asm2rtl[i](rtl_h, asm_context, rtl_batch)  # ASM → RTL
                asm_h = self.film_rtl2asm[i](asm_h, rtl_context, asm_batch)  # RTL → ASM
            else:
                # SSM-FiLM: 传递源节点特征，内部进行 SSM 处理
                rtl_h = self.film_asm2rtl[i](
                    rtl_h, asm_h, rtl_batch, asm_batch
                )  # ASM → RTL
                asm_h = self.film_rtl2asm[i](
                    asm_h, rtl_h, asm_batch, rtl_batch
                )  # RTL → ASM

        # 输出投影
        rtl_node = self.rtl_output(rtl_h)
        asm_node = self.asm_output(asm_h)

        # 图级池化
        rtl_graph = global_mean_pool(rtl_node, rtl_batch)
        asm_graph = global_mean_pool(asm_node, asm_batch)

        return rtl_node, rtl_edge_attr, rtl_graph, asm_node, asm_graph
