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
from torch_geometric.nn import GCNConv, MessagePassing, global_mean_pool
from torch_geometric.utils import softmax as pyg_softmax
from typing import Optional, Tuple
from .interaction import PerceiverCrossFusion

# EdgeType 索引常量（0-indexed，用于控制门控聚合和 Embedding 层）
# 对应 cdfg_rtl.data_types.EdgeType 枚举（值需减 1 转为 0-indexed）
EDGE_TYPE_DATA = 0
EDGE_TYPE_DATA_TRUE = 1
EDGE_TYPE_DATA_FALSE = 2
EDGE_TYPE_CONTROL = 3
EDGE_TYPE_CLOCK = 4
EDGE_TYPE_RESET = 5
EDGE_TYPE_ENABLE = 6

# NodeType 索引常量（0-indexed，对应 cdfg_rtl.data_types.NodeType 枚举）
NODE_TYPE_INPUT = 0
NODE_TYPE_OUTPUT = 1
NODE_TYPE_CONSTANT = 2
NODE_TYPE_COMBINATIONAL = 3
NODE_TYPE_SEQUENTIAL = 4
NODE_TYPE_MEMORY = 5
NODE_TYPE_MUX = 6
NODE_TYPE_ARITHMETIC = 7
NODE_TYPE_LOGIC = 8
NODE_TYPE_COMPARE = 9
NODE_TYPE_SHIFT = 10
NODE_TYPE_UNKNOWN = 11


# NodeType 0-indexed (0-11) → agg_group (0-9)
# INPUT(0)/OUTPUT(1)/CONSTANT(2) 共享 group 0
_NODE_TYPE_TO_AGG_GROUP = [
    0,  # INPUT
    0,  # OUTPUT
    0,  # CONSTANT
    1,  # COMBINATIONAL
    2,  # SEQUENTIAL
    3,  # MEMORY
    4,  # MUX  ← 实际走 gated_out
    5,  # ARITHMETIC
    6,  # LOGIC
    7,  # COMPARE
    8,  # SHIFT
    9,  # UNKNOWN
]
NUM_AGG_GROUPS = 10


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


class InstructionAdapter(nn.Module):
    """
    可训练适配器：为冻结的 instruction_encoding 补充任务适应能力。

    结构：LayerNorm → Linear → GELU → Linear（zero-init）+ 残差
    起步等价于恒等映射，旧 checkpoint 通过 strict=False 兼容加载。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        # zero-init 保证起步为恒等
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


class AsmNodeFeatureEncoder(nn.Module):
    """
    ASM 节点特征编码器

    将 node_type 和 instruction_encoding 编码为节点嵌入：
    - node_type: 通过 Embedding 层嵌入
    - instruction_encoding: 先经可训练 InstructionAdapter 适配，再线性投影
    - 融合方式: 加法融合
    """

    def __init__(
        self,
        num_node_types: int,
        instruction_dim: int,
        hidden_dim: int,
        use_adapter: bool = True,
    ):
        """
        Args:
            num_node_types: ASM 节点类型数量 (AsmNodeType 枚举)
            instruction_dim: 指令编码维度（由外部模型生成）
            hidden_dim: 隐藏层维度
            use_adapter: 是否启用 InstructionAdapter（默认 True）
        """
        super().__init__()
        self.node_type_embedding = nn.Embedding(num_node_types, hidden_dim)
        self.adapter = (
            InstructionAdapter(instruction_dim) if use_adapter else nn.Identity()
        )
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
        instr_emb = self.instruction_proj(self.adapter(instruction_encoding))  # [M, D]
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


class RTLGCNLayer(nn.Module):
    """Standard GCN propagation for RTL nodes, without RTL edge features."""

    uses_edge_features = False

    def __init__(self, hidden_dim: int, drop_path_rate: float = 0.1):
        super().__init__()
        self.conv = GCNConv(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop_path_rate = drop_path_rate

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        node_update = F.gelu(self.conv(x, edge_index))
        node_update = drop_path(node_update, self.drop_path_rate, self.training)
        return self.norm(x + node_update)


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

        # 类型感知软聚合选择器：每个 agg_group 学习 {sum, mean, max} 权重
        self.agg_selector = nn.Embedding(NUM_AGG_GROUPS, 3)
        nn.init.zeros_(self.agg_selector.weight)
        with torch.no_grad():
            self.agg_selector.weight[:, 0] = 2.0  # 初始偏向 sum
        self.register_buffer(
            "node_type_to_agg_group",
            torch.tensor(_NODE_TYPE_TO_AGG_GROUP, dtype=torch.long),
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

        # SEQUENTIAL 嵌套门控残差：[0]=reset 门, [1]=enable 门
        self.seq_gates = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
                for _ in range(2)
            ]
        )

        # ARITH/CMP/SHIFT 共享端口对 MLP，pair_type_emb 区分三类语义
        # idx: 0→ARITHMETIC, 1→COMPARE, 2→SHIFT
        self.pair_type_emb = nn.Embedding(3, hidden_dim)
        self.shared_pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # MEMORY 端口角色缩放：每个端口位置对应一个 D 维缩放向量，初始化为全 1
        self.mem_port_scale_emb = nn.Embedding(8, hidden_dim)
        nn.init.ones_(self.mem_port_scale_emb.weight)

        # GAT 风格注意力：per-edge 标量权重，让 edge_attr 内容影响默认聚合路径的贡献比例
        # 输入 cat([x_j, edge_attr]) [2D] → 标量 logit，per-node softmax 后用于加权求和
        self.attn_proj = nn.Linear(hidden_dim * 2, 1)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_type: Tensor,
        node_type: Tensor,
        target_port_idx: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        前向传播

        Args:
            x: [N, D] 节点特征
            edge_index: [2, E] 边索引
            edge_attr: [E, D] 边特征（已通过 RTLEdgeFeatureEncoder 编码）
            edge_type: [E] EdgeType 索引（用于控制门控聚合）
            node_type: [N] NodeType 0-indexed（用于类型感知聚合）
            target_port_idx: [E] 目标端口位置索引（可选，用于 ARITH/CMP/SHIFT/MEMORY）

        Returns:
            x: [N, D] 更新后的节点特征
            edge_attr: [E, D] 更新后的边特征
        """
        # 1. 计算 per-edge GAT 注意力权重（在 propagate 前用当前 x 和 edge_attr 计算）
        src = edge_index[0]
        fused_src_edge = torch.cat(
            [x[src], edge_attr], dim=-1
        )  # [E, 2D]，reuse in message()
        attn_logit = self.attn_proj(fused_src_edge)  # [E, 1]
        attn_w = pyg_softmax(attn_logit, edge_index[1], num_nodes=x.size(0))  # [E, 1]

        # 2. 消息传递 + 聚合（通过 propagate 调用 message 和 aggregate）
        node_out = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
            edge_type=edge_type,
            node_type=node_type,
            target_port_idx=target_port_idx,
            attn_w=attn_w,
            fused_src_edge=fused_src_edge,
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
        edge_type: Tensor,
        fused_src_edge: Tensor,
    ) -> Tensor:
        """计算边消息 φ(h_j, e_{j→i})"""
        is_ctrl = edge_type == EDGE_TYPE_CONTROL
        # 无条件计算所有边消息，用 torch.where 按类型选择（避免 .any() GPU-CPU 同步）
        ctrl_msg = self.ctrl_message_mlp(fused_src_edge).to(x_j.dtype)
        data_msg = self.data_message_mlp(fused_src_edge).to(x_j.dtype)
        return torch.where(is_ctrl.unsqueeze(-1), ctrl_msg, data_msg)

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        edge_type: Tensor,
        node_type: Tensor,
        target_port_idx: Optional[Tensor] = None,
        attn_w: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        """
        类型感知差异化聚合

        - MUX(6):        控制门控，α_true·m_true + α_false·m_false + m_other
        - SEQUENTIAL(4): 嵌套门控残差，模拟 reset > enable > data 优先级
        - MEMORY(5):     端口角色缩放，scaled_msg = port_scale_emb(port_idx) * msg
        - ARITHMETIC(7)/COMPARE(9)/SHIFT(10): 端口对 MLP，保留 portA/portB 顺序
        - 其他:          可学习 sum/mean/max 软混合（agg_selector）
        """
        num_nodes = dim_size if dim_size is not None else int(index.max()) + 1
        D = self.hidden_dim
        full_idx = index.unsqueeze(-1).expand(-1, D)

        is_ctrl = edge_type == EDGE_TYPE_CONTROL
        is_true = edge_type == EDGE_TYPE_DATA_TRUE
        is_false = edge_type == EDGE_TYPE_DATA_FALSE
        is_clock = edge_type == EDGE_TYPE_CLOCK
        is_reset = edge_type == EDGE_TYPE_RESET
        is_enable = edge_type == EDGE_TYPE_ENABLE
        is_data = edge_type == EDGE_TYPE_DATA

        def expand_idx(mask: Tensor) -> Tensor:
            return index[mask].unsqueeze(-1).expand(-1, D)

        agg_ctrl = inputs.new_zeros(num_nodes, D)
        agg_true = inputs.new_zeros(num_nodes, D)
        agg_false = inputs.new_zeros(num_nodes, D)
        agg_clock = inputs.new_zeros(num_nodes, D)
        agg_reset = inputs.new_zeros(num_nodes, D)
        agg_enable = inputs.new_zeros(num_nodes, D)
        agg_data = inputs.new_zeros(num_nodes, D)

        agg_ctrl.scatter_add_(0, expand_idx(is_ctrl), inputs[is_ctrl])
        agg_true.scatter_add_(0, expand_idx(is_true), inputs[is_true])
        agg_false.scatter_add_(0, expand_idx(is_false), inputs[is_false])
        agg_clock.scatter_add_(0, expand_idx(is_clock), inputs[is_clock])
        agg_reset.scatter_add_(0, expand_idx(is_reset), inputs[is_reset])
        agg_enable.scatter_add_(0, expand_idx(is_enable), inputs[is_enable])
        agg_data.scatter_add_(0, expand_idx(is_data), inputs[is_data])

        # MUX 门控聚合
        has_ctrl = inputs.new_zeros(num_nodes, dtype=torch.bool)
        has_ctrl.scatter_(0, index[is_ctrl], True)

        gate_logits = self.gate_mlp(agg_ctrl)
        gates = F.softmax(gate_logits, dim=-1)
        alpha_true = gates[:, 0:1]
        alpha_false = gates[:, 1:2]
        # clock/reset/enable/data 不参与互斥选择，直接加和旁路门控
        gated_out = (
            alpha_true * agg_true
            + alpha_false * agg_false
            + (agg_data + agg_clock + agg_reset + agg_enable)
        )

        # SEQUENTIAL: reset 同步覆盖 enable，enable 覆盖 data；乘法门保证高优先级激活时低优先级被完全屏蔽
        g_r = self.seq_gates[0](agg_reset)
        g_e = self.seq_gates[1](agg_enable)
        seq_out = g_r * agg_reset + (1 - g_r) * (g_e * agg_data + (1 - g_e) * agg_clock)

        # MEMORY 端口角色缩放（仅 DATA/ENABLE 边参与，排除 CONTROL/CLOCK 等干扰）
        if target_port_idx is not None:
            is_mem_edge = is_data | is_enable
            port_scale = self.mem_port_scale_emb(target_port_idx[is_mem_edge])
            scaled = inputs[is_mem_edge] * port_scale
            mem_out = inputs.new_zeros(num_nodes, D)
            mem_out.scatter_add_(0, expand_idx(is_mem_edge), scaled)
        else:
            mem_out = agg_data + agg_enable

        # ARITH/CMP/SHIFT 端口对分组
        if target_port_idx is not None:
            is_portA = target_port_idx == 0
            is_portB = target_port_idx == 1
            agg_portA = inputs.new_zeros(num_nodes, D)
            agg_portB = inputs.new_zeros(num_nodes, D)
            agg_portA.scatter_add_(0, expand_idx(is_portA), inputs[is_portA])
            agg_portB.scatter_add_(0, expand_idx(is_portB), inputs[is_portB])
        else:
            agg_portA = agg_data
            agg_portB = inputs.new_zeros(num_nodes, D)

        # 三路合一，减少 3 次独立 MLP forward 开销：stack [3N, 3D]，单次 forward，chunk 回各路
        portA_tiled = agg_portA.repeat(3, 1)
        portB_tiled = agg_portB.repeat(3, 1)
        type_tiled = self.pair_type_emb.weight.repeat_interleave(num_nodes, dim=0)
        pair_out = self.shared_pair_mlp(
            torch.cat([portA_tiled, portB_tiled, type_tiled], dim=-1)
        )
        arith_out, cmp_out, shift_out = pair_out.chunk(3, dim=0)

        # 默认软混合（LOGIC/COMB/其他）
        # all_sum 用于 all_mean/all_max 的分母计算；attn_sum 替代 all_sum 作为加权聚合
        all_sum = (
            agg_ctrl
            + agg_true
            + agg_false
            + agg_data
            + agg_clock
            + agg_reset
            + agg_enable
        )
        deg = inputs.new_zeros(num_nodes)
        deg.scatter_add_(0, index, torch.ones_like(index, dtype=inputs.dtype))
        all_mean = all_sum / deg.clamp(min=1).unsqueeze(-1)
        all_max = inputs.new_full((num_nodes, D), 0.0)
        all_max.scatter_reduce_(
            0,
            full_idx,
            inputs,
            reduce="amax",
            include_self=True,
        )

        attn_sum = inputs.new_zeros(num_nodes, D)
        attn_sum.scatter_add_(0, full_idx, inputs * attn_w)

        agg_group = self.node_type_to_agg_group[node_type]
        agg_w = F.softmax(self.agg_selector(agg_group), dim=-1)
        type_out = (
            agg_w[:, 0:1] * attn_sum
            + agg_w[:, 1:2] * all_mean
            + agg_w[:, 2:3] * all_max
        )

        # 路由链（内到外，MUX 最高优先级）
        result = type_out
        result = torch.where(
            (node_type == NODE_TYPE_SHIFT).unsqueeze(-1), shift_out, result
        )
        result = torch.where(
            (node_type == NODE_TYPE_COMPARE).unsqueeze(-1), cmp_out, result
        )
        result = torch.where(
            (node_type == NODE_TYPE_ARITHMETIC).unsqueeze(-1), arith_out, result
        )
        result = torch.where(
            (node_type == NODE_TYPE_MEMORY).unsqueeze(-1), mem_out, result
        )
        result = torch.where(
            (node_type == NODE_TYPE_SEQUENTIAL).unsqueeze(-1), seq_out, result
        )
        result = torch.where(has_ctrl.unsqueeze(-1), gated_out, result)  # MUX
        return result

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


class DualGraphEncoder(nn.Module):
    """
    双向 Perceiver 注入式双图编码器

    架构：每层 GNN 后，ASM 和 RTL 通过双向 PerceiverCrossFusion 相互注入。

    特性：
    - 节点级双向 cross-attn，细粒度 ASM ↔ RTL 对应
    - K 个 latent token 作为信息瓶颈，O((N_rtl+N_asm)·K·D) 复杂度
    - 双向调用使用快照，避免链式污染
    - ASM 节点编码含可训练 Adapter
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
        max_ports: int = 8,
        # ASM 特征配置
        num_asm_node_types: int = 22,
        num_asm_edge_types: int = 10,
        asm_instruction_dim: int = 256,
        use_asm_adapter: bool = True,
        # Perceiver 融合配置
        perceiver_num_latents: int = 16,
        perceiver_num_heads: int = 4,
        enable_cross_fusion: bool = True,
        rtl_layer_cls: type[nn.Module] = ControlGatedGNNLayer,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.enable_cross_fusion = enable_cross_fusion

        # RTL 编码器
        self.rtl_node_encoder = RTLNodeFeatureEncoder(num_cell_types, hidden_dim)
        self.rtl_edge_encoder = (
            RTLEdgeFeatureEncoder(num_edge_types, hidden_dim, max_ports)
            if getattr(rtl_layer_cls, "uses_edge_features", True)
            else None
        )

        # ASM 编码器（含可训练 Adapter）
        self.asm_node_encoder = AsmNodeFeatureEncoder(
            num_asm_node_types, asm_instruction_dim, hidden_dim, use_asm_adapter
        )
        self.asm_edge_encoder = AsmEdgeFeatureEncoder(num_asm_edge_types, hidden_dim)

        # GNN 层（逐层增加 drop_path）
        drop_rates = [
            dropout * i / (num_layers - 1) if num_layers > 1 else 0.0
            for i in range(num_layers)
        ]
        self.rtl_layers = nn.ModuleList(
            [rtl_layer_cls(hidden_dim, drop_rates[i]) for i in range(num_layers)]
        )
        self.asm_layers = nn.ModuleList(
            [GNNLayer(hidden_dim, drop_rates[i]) for i in range(num_layers)]
        )

        # Baseline 不实例化融合层，确保参数量和 checkpoint 中均无融合参数。
        if enable_cross_fusion:
            self.fusion_asm2rtl = nn.ModuleList(
                [
                    PerceiverCrossFusion(
                        hidden_dim,
                        perceiver_num_latents,
                        perceiver_num_heads,
                        num_layers,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.fusion_rtl2asm = nn.ModuleList(
                [
                    PerceiverCrossFusion(
                        hidden_dim,
                        perceiver_num_latents,
                        perceiver_num_heads,
                        num_layers,
                    )
                    for _ in range(num_layers)
                ]
            )

        # 输出投影
        self.rtl_output = nn.Linear(hidden_dim, output_dim)
        self.asm_output = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        # RTL 图
        rtl_edge_index: torch.Tensor,
        rtl_node_cell_type: torch.Tensor,
        rtl_node_width: torch.Tensor,
        rtl_edge_type: torch.Tensor,
        rtl_node_type: torch.Tensor,
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
        Returns:
            rtl_node:     [N1, D] RTL 节点嵌入
            rtl_edge_attr:[E1, D] RTL 边嵌入
            rtl_graph:    [B, D]  RTL 图级嵌入
            asm_node:     [N2, D] ASM 节点嵌入
            asm_graph:    [B, D]  ASM 图级嵌入
        """
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

        rtl_h = self.rtl_node_encoder(rtl_node_cell_type, rtl_node_width)
        asm_h = self.asm_node_encoder(asm_node_type, asm_instruction_encoding)

        if self.rtl_edge_encoder is None:
            rtl_edge_attr = rtl_h.new_zeros((rtl_edge_index.shape[1], self.hidden_dim))
        else:
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
            if self.rtl_edge_encoder is None:
                rtl_h = self.rtl_layers[i](rtl_h, rtl_edge_index)
            else:
                rtl_h, rtl_edge_attr = self.rtl_layers[i](
                    rtl_h,
                    rtl_edge_index,
                    rtl_edge_attr,
                    rtl_edge_type,
                    rtl_node_type,
                    rtl_edge_target_port_idx,
                )

            if self.enable_cross_fusion:
                # 双向 Perceiver 注入：快照避免链式污染
                rtl_in, asm_in = rtl_h, asm_h
                rtl_h = self.fusion_asm2rtl[i](
                    rtl_in, asm_in, rtl_batch, asm_batch
                )  # ASM → RTL
                asm_h = self.fusion_rtl2asm[i](
                    asm_in, rtl_in, asm_batch, rtl_batch
                )  # RTL → ASM

        rtl_node = self.rtl_output(rtl_h)
        asm_node = self.asm_output(asm_h)
        rtl_graph = global_mean_pool(rtl_node, rtl_batch)
        asm_graph = global_mean_pool(asm_node, asm_batch)

        return rtl_node, rtl_edge_attr, rtl_graph, asm_node, asm_graph


# Backward-compatible import name for callers that use the fusion-default encoder.
PerceiverDualEncoder = DualGraphEncoder
