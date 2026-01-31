# 节点-边联合更新图神经网络的数学分析

## 1. 数学形式化定义

本网络实现节点特征和边特征的联合更新，可以用以下数学公式表示：

### 阶段 1：节点特征更新（控制门控聚合）

对于 MUX 节点，传统的置换不变聚合（sum/mean/max）无法表达控制边与数据边的**角色差异**和**互斥选择**关系。

**MUX 的物理行为**：
- 控制边（control）决定选择哪条数据路径
- 数据边（data_true, data_false）是互斥的，同一时刻只有一条有效
- `sel=1` 时选择 `data_true`，`sel=0` 时选择 `data_false`

**控制门控聚合公式**：

$$\mathbf{h}_i^{(k)} = \gamma^{(k)} \left( \mathbf{h}_i^{(k-1)}, \mathbf{m}_{\text{gated}}^{(k)} \right)$$

其中门控消息计算为：

$$\mathbf{m}_{\text{gated}} = \alpha_{\text{true}} \cdot \mathbf{m}_{\text{true}} + \alpha_{\text{false}} \cdot \mathbf{m}_{\text{false}} + \mathbf{m}_{\text{other}}$$

门控权重由控制边决定：

$$[\alpha_{\text{true}}, \alpha_{\text{false}}] = \text{softmax}\left( f_{\text{gate}}\left( \mathbf{m}_{\text{ctrl}} \right) \right)$$

各类消息的计算（支持多条同类型边）：

$$\mathbf{m}_{\text{true}} = \bigoplus_{j \in \mathcal{N}_{\text{true}}^{-}(i)} \phi(\mathbf{h}_j, \mathbf{e}_{j \to i})$$
$$\mathbf{m}_{\text{false}} = \bigoplus_{j \in \mathcal{N}_{\text{false}}^{-}(i)} \phi(\mathbf{h}_j, \mathbf{e}_{j \to i})$$
$$\mathbf{m}_{\text{ctrl}} = \bigoplus_{j \in \mathcal{N}_{\text{ctrl}}^{-}(i)} \phi_{\text{ctrl}}(\mathbf{h}_j, \mathbf{e}_{j \to i})$$
$$\mathbf{m}_{\text{other}} = \bigoplus_{j \in \mathcal{N}_{\text{other}}^{-}(i)} \phi(\mathbf{h}_j, \mathbf{e}_{j \to i})$$

其中：
- $\mathbf{h}_i^{(k)} \in \mathbb{R}^{d_h}$：第 $k$ 层节点 $i$ 的特征
- $\mathcal{N}_{\text{true}}^{-}(i)$：**所有** data_true 边的源节点集合（可能有多个）
- $\mathcal{N}_{\text{false}}^{-}(i)$：**所有** data_false 边的源节点集合（可能有多个）
- $\mathcal{N}_{\text{ctrl}}^{-}(i)$：control 边的源节点集合
- $\mathcal{N}_{\text{other}}^{-}(i)$：其他边（CLOCK, RESET, ENABLE 等）的源节点集合
- $\phi^{(k)}$：数据边消息函数（MLP）
- $\phi_{\text{ctrl}}^{(k)}$：控制边消息函数（MLP）
- $f_{\text{gate}}$：门控函数（MLP），输出 2 维向量
- $\alpha_{\text{true}} + \alpha_{\text{false}} = 1$：互斥选择约束
- $\gamma^{(k)}$：节点更新函数（MLP）
- $\bigoplus$：置换不变聚合函数（sum），用于聚合同类型的多条边

**多数据边场景**：当一个 MUX 有多个 data_true 或 data_false 边时（如多位宽信号或数据路径复制），先对同类型边进行 sum 聚合，再进行门控加权。这确保了：
- 同一分支的所有输入信号被**整体选中或整体忽略**
- 门控权重作用于**聚合后的分支表示**，而非单条边

**对于非 MUX 节点**（没有 control 边），回退到标准聚合：

$$\mathbf{h}_i^{(k)} = \gamma^{(k)} \left( \mathbf{h}_i^{(k-1)}, \bigoplus_{j \in \mathcal{N}^{-}(i)} \phi(\mathbf{h}_j, \mathbf{e}_{j \to i}) \right)$$

### 阶段 2：边特征更新

$$\mathbf{e}_{j \to i}^{(k)} = \psi^{(k)} \left( \mathbf{h}_j^{(k)}, \mathbf{e}_{j \to i}^{(k-1)} \right)$$

其中：
- $\psi^{(k)}: \mathbb{R}^{d_h} \times \mathbb{R}^{d_e} \to \mathbb{R}^{d_e}$：边更新函数（MLP）
- 边更新**仅依赖源节点**（驱动端），不依赖目标节点（负载端）

**设计理由**：在 RTL 电路中，边（Wire/Net）的状态由驱动它的信号源决定，目标节点只是被动接收者。

## 2. 与现有 GNN 变体的关系

| 模型 | 节点更新 | 边更新 | 特点 |
|------|---------|--------|------|
| GCN | $\sum_j \mathbf{h}_j$ | ❌ | 无边特征 |
| GAT | $\sum_j \alpha_{ij} \mathbf{h}_j$ | ❌ | 注意力权重 |
| GINEConv | $\sum_j (\mathbf{h}_j + \mathbf{e}_{ji})$ | ❌ | 边特征参与但不更新 |
| MPNN (NNConv) | $\sum_j \mathbf{W}_{e_{ji}} \mathbf{h}_j$ | ❌ | 边特征调制权重 |
| **本网络** | 控制门控聚合 | ✅ $\psi(\mathbf{h}_j, \mathbf{e}_{ji})$ | **MUX 互斥选择 + 边仅依赖源节点** |

## 3. RTL 代码执行语义的模拟

本网络结构**天然适合模拟 RTL 电路中信号的传播**：

### 3.1 节点 = 电路组件（Cell）

| RTL 概念 | 图节点 | 特征含义 |
|----------|--------|----------|
| MUX | 控制流节点 | 选择状态、分支概率 |
| DFF/寄存器 | 时序节点 | 时钟域、复位状态 |
| 组合逻辑 | 运算节点 | 操作类型编码 |
| 端口 | I/O 节点 | 驱动/负载特性 |

### 3.2 边 = 信号连接（Wire/Net）

| RTL 概念 | 图边 | 特征含义 |
|----------|------|----------|
| 数据线 | DATA 边 | 通用数据流 |
| MUX 真分支 | DATA_TRUE 边 | B 端口数据（S=1 时选择） |
| MUX 假分支 | DATA_FALSE 边 | A 端口数据（S=0 时选择） |
| 控制信号 | CONTROL 边 | 使能条件、分支索引 |
| 时钟 | CLOCK 边 | 时钟域标识 |
| 复位 | RESET 边 | 复位类型（同步/异步） |
| 使能 | ENABLE 边 | 使能信号 |

### 3.3 消息传递 ≈ 信号传播

**关键洞察**：RTL 中信号沿着**有向边**从驱动端传播到负载端，这与本设计完美匹配：

```
信号传播方向：
  Driver (node j) ──edge(j→i)──> Load (node i)

GNN 消息传递：
  h_j + e_{j→i} ──message──> aggregate at i ──update──> h_i'
```

### 3.4 边更新的物理意义

边特征更新模拟了**信号在连接上的传播状态**：

$$\mathbf{e}_{j \to i}^{(k)} = \psi(\mathbf{h}_j^{(k)}, \mathbf{e}_{j \to i}^{(k-1)})$$

| 更新依赖 | 物理意义 |
|----------|----------|
| $\mathbf{h}_j^{(k)}$（源节点） | 驱动信号的**当前状态**（值、有效性） |
| $\mathbf{e}_{j \to i}^{(k-1)}$（边自身） | 连接的**固有属性**（类型、宽度、覆盖率标签） |

**为什么不依赖目标节点 $\mathbf{h}_i$？**

在 RTL 电路中：
- **边 = 导线（Wire）**：导线只是传输媒介，其状态由驱动端决定
- **因果方向**：信号从 j 流向 i，边的激活状态是 j 的"输出"
- **覆盖率语义**：边是否被覆盖取决于驱动信号是否被激活，与负载如何处理无关

```verilog
// 示例：边 e1 的覆盖状态仅取决于 in_false，与 MUX 无关
assign out = sel ? in_true : in_false;
//           ↑        ↑         ↑
//          e3       e2        e1
```

这可以建模：
- **覆盖率传播**：已覆盖的输入 → 边变为"已激活"
- **可达性分析**：信号能否从输入到达该边
- **时序约束**：跨时钟域的边需要同步处理

## 4. 与 RTL 执行的精确对应

以一个 MUX 节点的更新过程为例：

```verilog
assign out = sel ? in_true : in_false;  // RTL 三元运算
```

综合后的 CDFG：
```
in_false (A) ──data_false──┐
                           ├──> MUX ──> out
in_true  (B) ──data_true───┘
                    ↑
        sel ────control────
```

**GNN 层的模拟（控制门控聚合）**：

```
第 k 层更新：

1. 分类计算消息（按边的 coverage_type）：
   m_ctrl  = φ_ctrl(h_sel, e_ctrl)        # 控制消息
   m_true  = φ(h_in_true,  e_data_true)   # 真分支消息
   m_false = φ(h_in_false, e_data_false)  # 假分支消息

2. 生成门控权重（模拟 sel 的选择）：
   [α_true, α_false] = softmax(f_gate(m_ctrl))

   # 理想情况：
   # sel=1 时: α_true ≈ 1, α_false ≈ 0 → 选择 in_true
   # sel=0 时: α_true ≈ 0, α_false ≈ 1 → 选择 in_false

3. 门控聚合：
   m_gated = α_true · m_true + α_false · m_false

4. 节点更新：
   h_MUX' = γ(h_MUX, m_gated)

5. 边更新（仅用源节点特征）：
   e_data_false' = ψ(h_in_false', e_data_false)
   e_data_true'  = ψ(h_in_true',  e_data_true)
   e_ctrl'       = ψ(h_sel',      e_ctrl)
```

### 4.1 多数据边场景

当一个 MUX 有多个同类型输入边时（数据路径复制）：

```verilog
// 一个 IF 条件驱动多个输出
always_comb begin
    if (sel) begin
        out1 = a;
        out2 = b;
    end else begin
        out1 = c;
        out2 = d;
    end
end
```

综合后的 CDFG（多数据边）：
```
a ──data_true───┐
b ──data_true───┼──> MUX
c ──data_false──┤
d ──data_false──┘
        ↑
sel ──control──
```

**聚合过程**：

```
1. 分类聚合消息（同类型边先 sum）：
   m_ctrl  = φ_ctrl(h_sel, e_ctrl)
   m_true  = φ(h_a, e_a) + φ(h_b, e_b)      # 聚合所有 data_true
   m_false = φ(h_c, e_c) + φ(h_d, e_d)      # 聚合所有 data_false

2. 生成门控权重：
   [α_true, α_false] = softmax(f_gate(m_ctrl))

3. 门控聚合（作用于聚合后的分支表示）：
   m_gated = α_true · m_true + α_false · m_false

   # 当 sel=1 时：a 和 b 的信息整体通过
   # 当 sel=0 时：c 和 d 的信息整体通过
```

**语义保证**：同一分支的所有输入被**整体选中或整体忽略**，符合 RTL 的物理行为。

### 4.2 EdgeType 

#### EdgeType（结构类型，CDFG 提取阶段设置）

| EdgeType | 索引 | 含义 | MUX 端口 |
|----------|------|------|----------|
| `DATA` | 0 | 通用数据流 | - |
| `DATA_TRUE` | 1 | MUX 真分支数据流 | B 端口 |
| `DATA_FALSE` | 2 | MUX 假分支数据流 | A 端口 |
| `CONTROL` | 3 | 控制流 | S 端口 |
| `CLOCK` | 4 | 时钟 | CLK 端口 |
| `RESET` | 5 | 复位 | RST 端口 |
| `ENABLE` | 6 | 使能 | EN 端口 |

`EdgeType` 提供结构信息，控制门控聚合使用 

## 5. 覆盖率预测的数学解释

本网络特别适合**边级覆盖率预测**任务：

$$\hat{y}_{j \to i} = \sigma \left( \text{MLP}(\mathbf{e}_{j \to i}^{(K)}) \right)$$

**信息流分析**：

经过 $K$ 层传播后，边特征 $\mathbf{e}_{j \to i}^{(K)}$ 编码了：

1. **局部结构**：源节点的类型和属性
2. **上下文信息**：$K$ 跳邻域内的拓扑和数据流模式
3. **全局可达性**：从输入端口到该边的路径信息

这正是判断一条边是否被测试激励覆盖所需的信息！

## 6. 数学性质分析

### 6.1 表达能力

**定理**（非正式）：本网络至少与 1-WL 测试等价，且由于边更新机制，可以区分某些 1-WL 无法区分的图。

**证明思路**：边特征更新增加了边的"身份信息"，使得即使两个节点有相同的邻居多重集，如果连接边的特征不同，更新后的状态也不同。

### 6.2 过平滑问题

边更新机制有助于**缓解过平滑**：

- 传统 GNN：深层节点特征趋于相同
- 本网络：边特征保持了额外的区分信息

$$\text{Var}(\mathbf{e}^{(K)}) \geq \text{Var}(\mathbf{h}^{(K)})$$

边特征的异质性可以"锚定"节点特征，减缓收敛到平凡解的速度。

## 7. PyG 实现框架

### 7.1 控制门控聚合层

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing

# EdgeType 索引常量（用于控制门控聚合）
EDGE_TYPE_DATA = 0
EDGE_TYPE_DATA_TRUE = 1
EDGE_TYPE_DATA_FALSE = 2
EDGE_TYPE_CONTROL = 3
EDGE_TYPE_CLOCK = 4
EDGE_TYPE_RESET = 5
EDGE_TYPE_ENABLE = 6

class ControlGatedGNNLayer(MessagePassing):
    """
    控制门控聚合的 GNN 层，精确模拟 MUX 的互斥选择行为

    对于 MUX 节点（有 control 边）：
    - 控制边生成门控权重 [α_true, α_false]
    - 门控权重决定 data_true 和 data_false 的贡献比例
    - softmax 确保互斥选择（权重和为 1）

    对于非 MUX 节点（无 control 边）：
    - 回退到标准 sum 聚合
    """

    def __init__(self, hidden_dim: int, num_edge_types: int = 7):
        super().__init__(aggr=None)  # 自定义聚合
        self.hidden_dim = hidden_dim

        # φ: 数据边消息函数
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

        # EdgeType 嵌入（融入边特征）
        self.edge_type_embedding = nn.Embedding(num_edge_types, hidden_dim)

        # γ: 节点更新函数
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # ψ: 边更新函数（仅依赖源节点）
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: Tensor,                    # [N, D] 节点特征
        edge_index: Tensor,           # [2, E] 边索引
        edge_attr: Tensor,            # [E, D] 边特征
        edge_type: Tensor,            # [E] EdgeType 索引
    ) -> tuple[Tensor, Tensor]:
        num_nodes = x.size(0)
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        # 融合 EdgeType 嵌入到边特征
        edge_type_emb = self.edge_type_embedding(edge_type)
        edge_attr_enhanced = edge_attr + edge_type_emb

        # 计算消息
        x_j = x[source_nodes]
        is_ctrl = (edge_type == EDGE_TYPE_CONTROL)

        messages = torch.zeros(edge_index.size(1), self.hidden_dim, device=x.device)

        # 控制边消息
        if is_ctrl.any():
            ctrl_input = torch.cat([x_j[is_ctrl], edge_attr_enhanced[is_ctrl]], dim=-1)
            messages[is_ctrl] = self.ctrl_message_mlp(ctrl_input)

        # 数据边消息
        is_data = ~is_ctrl
        if is_data.any():
            data_input = torch.cat([x_j[is_data], edge_attr_enhanced[is_data]], dim=-1)
            messages[is_data] = self.data_message_mlp(data_input)

        # 控制门控聚合
        node_out = self._gated_aggregate(
            messages, target_nodes, edge_type, num_nodes
        )
        x = self.node_norm(x + self.node_mlp(torch.cat([x, node_out], dim=-1)))

        # 边更新（仅依赖源节点）
        x_j_new = x[source_nodes]
        edge_out = self.edge_mlp(torch.cat([x_j_new, edge_attr], dim=-1))
        edge_attr = self.edge_norm(edge_attr + edge_out)

        return x, edge_attr

    def _gated_aggregate(
        self,
        messages: Tensor,           # [E, D]
        target_nodes: Tensor,       # [E]
        edge_type: Tensor,          # [E] EdgeType 索引
        num_nodes: int,
    ) -> Tensor:
        """向量化的控制门控聚合"""
        device = messages.device

        # 创建各类型的 mask（基于 EdgeType）
        is_ctrl = (edge_type == EDGE_TYPE_CONTROL)
        is_true = (edge_type == EDGE_TYPE_DATA_TRUE)
        is_false = (edge_type == EDGE_TYPE_DATA_FALSE)
        is_other = ~(is_ctrl | is_true | is_false)

        # 分别聚合各类消息
        agg_ctrl = messages.new_zeros(num_nodes, self.hidden_dim)
        agg_true = messages.new_zeros(num_nodes, self.hidden_dim)
        agg_false = messages.new_zeros(num_nodes, self.hidden_dim)
        agg_other = messages.new_zeros(num_nodes, self.hidden_dim)

        idx_expand = lambda idx: idx.unsqueeze(-1).expand(-1, self.hidden_dim)

        if is_ctrl.any():
            agg_ctrl.scatter_add_(0, idx_expand(target_nodes[is_ctrl]), messages[is_ctrl])
        if is_true.any():
            agg_true.scatter_add_(0, idx_expand(target_nodes[is_true]), messages[is_true])
        if is_false.any():
            agg_false.scatter_add_(0, idx_expand(target_nodes[is_false]), messages[is_false])
        if is_other.any():
            agg_other.scatter_add_(0, idx_expand(target_nodes[is_other]), messages[is_other])

        # 检测哪些节点有 control 边（MUX 节点）
        has_ctrl = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        if is_ctrl.any():
            has_ctrl.scatter_(0, target_nodes[is_ctrl], True)

        # 生成门控权重
        gate_logits = self.gate_mlp(agg_ctrl)  # [N, 2]
        gates = F.softmax(gate_logits, dim=-1)
        gate_true = gates[:, 0:1]   # [N, 1]
        gate_false = gates[:, 1:2]  # [N, 1]

        # MUX 节点：门控加权
        gated_out = gate_true * agg_true + gate_false * agg_false + agg_other

        # 非 MUX 节点：标准聚合
        standard_out = agg_ctrl + agg_true + agg_false + agg_other

        # 选择输出
        return torch.where(has_ctrl.unsqueeze(-1), gated_out, standard_out)
```

### 7.2 简化版本（无门控，用于对比）

```python
class RTLConv(MessagePassing):
    """节点-边联合更新的 GNN 层（标准聚合版本）"""

    def __init__(self, hidden_dim: int):
        super().__init__(aggr='sum', flow='source_to_target')

        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # ψ: 边更新函数 - 仅依赖源节点
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        # 节点更新
        node_out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        x = self.node_norm(x + node_out)

        # 边更新（仅使用源节点特征）
        edge_out = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)
        edge_attr = self.edge_norm(edge_attr + edge_out)

        return x, edge_attr

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return self.message_mlp(torch.cat([x_j, edge_attr], dim=-1))

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        return self.node_mlp(torch.cat([x, aggr_out], dim=-1))

    def edge_update(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        """边更新: ψ(h_j, e_{j→i}) - 仅依赖源节点"""
        return self.edge_mlp(torch.cat([x_j, edge_attr], dim=-1))
```

## 8. 总结

| 设计决策 | 数学意义 | RTL 对应 |
|----------|----------|----------|
| 控制门控聚合 | $\alpha_{\text{true}} \cdot \mathbf{m}_{\text{true}} + \alpha_{\text{false}} \cdot \mathbf{m}_{\text{false}}$ | MUX 的互斥选择行为 |
| 门控权重由控制边生成 | $\boldsymbol{\alpha} = \text{softmax}(f_{\text{gate}}(\mathbf{m}_{\text{ctrl}}))$ | sel 信号决定数据通路 |
| 消息依赖边特征 | $\phi(\mathbf{h}_j, \mathbf{e}_{ji})$ | 信号类型影响传播方式 |
| 先更新节点再更新边 | 顺序依赖 | 节点状态决定输出信号 |
| 边更新仅依赖源节点 | $\psi(\mathbf{h}_j, \mathbf{e}_{ji})$ | 边状态由驱动端决定 |
| 非 MUX 节点回退 | 标准 sum 聚合 | 普通组合逻辑 |

本网络设计在数学上是**良定义的消息传递框架**，在物理上**精确模拟了 RTL 电路的信号传播语义**，是进行覆盖率预测的理想选择。

## 9. 整图回归任务的适用性分析

### 9.1 当前网络的局限性

本网络设计重点在**边级特征学习**，但整图回归需要将所有信息压缩为**单一标量/向量**：

$$\hat{y}_{\text{graph}} = f(\{\mathbf{h}_i^{(K)}\}_{i \in V}, \{\mathbf{e}_{j \to i}^{(K)}\}_{(j,i) \in E})$$

**核心问题**：如何有效聚合节点和边特征？

### 9.2 整图回归的信息流对比

| 任务类型 | 信息流向 | 输出 | 本网络适配度 |
|----------|----------|------|-------------|
| 边分类 | 局部 → 边 | $\hat{y}_{e} \in \{0,1\}$ | ✅ 天然适合 |
| 节点分类 | 邻域 → 节点 | $\hat{y}_{v} \in \mathbb{R}^C$ | ✅ 适合 |
| **整图回归** | 全图 → 标量 | $\hat{y}_{G} \in \mathbb{R}$ | ⚠️ 需要扩展 |

### 9.3 扩展方案：双池化读出（Dual Readout）

为了适配整图回归，需要添加**图级读出层**：

$$\mathbf{z}_G = \text{READOUT}\left(\{\mathbf{h}_i^{(K)}\}, \{\mathbf{e}_{j \to i}^{(K)}\}\right)$$

#### 方案 A：简单拼接池化

$$\mathbf{z}_G = \left[ \bigoplus_{i \in V} \mathbf{h}_i^{(K)} \; \| \; \bigoplus_{(j,i) \in E} \mathbf{e}_{j \to i}^{(K)} \right]$$

```python
# 节点池化
h_graph = global_mean_pool(x, batch)  # [B, d_h]

# 边池化（需要 edge_batch 索引）
e_graph = global_mean_pool(edge_attr, edge_batch)  # [B, d_e]

# 拼接
z_graph = torch.cat([h_graph, e_graph], dim=-1)  # [B, d_h + d_e]
```

#### 方案 B：注意力加权池化

$$\mathbf{z}_G = \sum_{i \in V} \alpha_i \mathbf{h}_i^{(K)} + \sum_{(j,i) \in E} \beta_{ji} \mathbf{e}_{j \to i}^{(K)}$$

其中注意力权重：

$$\alpha_i = \frac{\exp(\text{MLP}_v(\mathbf{h}_i))}{\sum_{i'} \exp(\text{MLP}_v(\mathbf{h}_{i'}))}$$

#### 方案 C：层次池化（推荐用于大图）

```
原始图 → Coarsening → 粗化图 → Coarsening → ... → 单节点
```

### 9.4 RTL 覆盖率整图回归的物理意义

整图回归目标：**预测整体覆盖率百分比**

$$\hat{y}_G = \frac{\text{已覆盖边数}}{\text{总边数}} \approx \frac{1}{|E|} \sum_{e \in E} \sigma(\mathbf{e}^{(K)})$$

**关键洞察**：整图覆盖率 ≈ 边级预测的平均值

这意味着本网络**天然支持**这种形式的整图回归：

$$\hat{y}_G = \text{MLP}\left( \frac{1}{|E|} \sum_{(j,i) \in E} \mathbf{e}_{j \to i}^{(K)} \right)$$

### 9.5 数学分析：边特征池化的充分性

**定理**：对于 RTL 覆盖率预测，边特征池化比节点特征池化信息更充分。

**证明思路**：
1. 覆盖率标签定义在边上，不在节点上
2. 边特征 $\mathbf{e}_{j \to i}^{(K)}$ 编码了源节点信息（通过 $\psi$ 函数）
3. 因此：$I(\mathbf{e}^{(K)}; y_G) \geq I(\mathbf{h}^{(K)}; y_G)$

### 9.6 推荐的整图回归架构

```python
from torch_geometric.nn import global_mean_pool

class RTLGraphRegressor(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers):
        super().__init__()

        # 多层 RTLConv
        self.convs = nn.ModuleList([
            RTLConv(node_dim, edge_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # 边特征池化 + 投影
        self.edge_pool_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 可选：节点特征池化
        self.node_pool_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 最终回归头
        self.regressor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 覆盖率在 [0, 1]
        )

    def forward(self, x, edge_index, edge_attr, batch, edge_batch):
        # 消息传递
        for conv in self.convs:
            x, edge_attr = conv(x, edge_index, edge_attr)

        # 双池化读出
        h_graph = global_mean_pool(self.node_pool_mlp(x), batch)
        e_graph = global_mean_pool(self.edge_pool_mlp(edge_attr), edge_batch)

        # 拼接并回归
        z_graph = torch.cat([h_graph, e_graph], dim=-1)
        return self.regressor(z_graph)
```

### 9.7 整图回归适用性总结

| 问题 | 答案 |
|------|------|
| 网络是否适合整图回归？ | ⚠️ 需要添加池化层 |
| 最佳池化策略？ | **边特征为主** + 节点特征辅助 |
| 物理意义？ | 整图覆盖率 ≈ 边级覆盖率的聚合 |
| 信息是否充分？ | ✅ 边特征已编码节点信息 |

**结论**：本网络**可以适配整图回归**，但需要：
1. 添加 `edge_batch` 索引支持边级池化
2. 使用**边特征为主导**的双池化读出
3. 对于覆盖率预测，边池化的物理意义更直接
