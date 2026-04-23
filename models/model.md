# 节点-边联合更新图神经网络的数学分析

## 1. 数学形式化定义

本网络实现节点特征和边特征的联合更新，可以用以下数学公式表示：

### 阶段 1：节点特征更新（类型感知差异化聚合）

传统的置换不变聚合（sum/mean/max）无法表达 RTL 电路中不同类型节点的角色差异：
- **MUX**：控制边与数据边互斥选择关系
- **SEQUENTIAL**：reset > enable > data 的信号优先级
- **MEMORY**：不同端口（地址/数据/使能）角色各异
- **ARITHMETIC/COMPARE/SHIFT**：操作数对 (A, B) 顺序敏感

因此本网络为各节点类型实现专用聚合逻辑，统一通过路由链按优先级选择：

$$\mathbf{m}_i = \text{Route}(\text{node\_type}_i, \{\mathbf{m}_{i,\text{MUX}}, \mathbf{m}_{i,\text{SEQ}}, \mathbf{m}_{i,\text{MEM}}, \mathbf{m}_{i,\text{pair}}, \mathbf{m}_{i,\text{soft}}\})$$

---

### 1.1 MUX 控制门控聚合

**物理行为**：控制边决定选择哪条数据路径，data_true/data_false 互斥。

$$\mathbf{m}_{\text{MUX}} = \alpha_{\text{true}} \cdot \mathbf{m}_{\text{true}} + \alpha_{\text{false}} \cdot \mathbf{m}_{\text{false}} + (\mathbf{m}_{\text{data}} + \mathbf{m}_{\text{clock}} + \mathbf{m}_{\text{reset}} + \mathbf{m}_{\text{enable}})$$

门控权重由控制边聚合消息决定：

$$[\alpha_{\text{true}}, \alpha_{\text{false}}] = \text{softmax}\left( f_{\text{gate}}\left( \mathbf{m}_{\text{ctrl}} \right) \right)$$

**注意**：clock/reset/enable/data 不参与互斥选择，直接加和旁路门控。

---

### 1.2 SEQUENTIAL 嵌套门控残差

**物理行为**：RTL 时序逻辑的优先级严格为 reset > enable > data。软加权 $\sum w_t m_t$ 无法表达高优先级激活时低优先级被完全屏蔽的乘法关系。

$$g_r = \sigma(\text{MLP}(\mathbf{m}_{\text{reset}})) \in (0,1)^d$$
$$g_e = \sigma(\text{MLP}(\mathbf{m}_{\text{enable}})) \in (0,1)^d$$
$$\mathbf{m}_{\text{SEQ}} = g_r \odot \mathbf{m}_{\text{reset}} + (1 - g_r) \odot \left( g_e \odot \mathbf{m}_{\text{data}} + (1 - g_e) \odot \mathbf{m}_{\text{clock}} \right)$$

当 $g_r \approx \mathbf{1}$ 时，enable/data/clock 路径被完全屏蔽，精确模拟复位覆盖使能的物理行为。

---

### 1.3 MEMORY 端口角色缩放

**物理行为**：MEMORY 节点的端口（如地址、写数据、读数据、使能）在功能上角色完全不同，不可等权聚合。

$$\mathbf{m}_{\text{MEM}} = \sum_{e \in \mathcal{N}^-(i),\, t_e \in \{\text{DATA},\text{ENABLE}\}} \mathbf{s}_{p_e} \odot \mathbf{m}_e$$

其中 $\mathbf{s}_{p_e} = \text{Embedding}_{\text{port}}(p_e) \in \mathbb{R}^d$ 是目标端口位置索引 $p_e$ 对应的缩放向量（初始化为全 1 = 恒等缩放）。

**注意**：仅 DATA 和 ENABLE 类型的边参与端口角色缩放聚合；CONTROL、CLOCK、DATA_TRUE、DATA_FALSE 等边被排除，避免控制/时钟信号以任意端口索引混入 MEMORY 聚合路径。

---

### 1.4 ARITHMETIC/COMPARE/SHIFT 端口对 MLP

**物理行为**：这类节点天然有两个操作数端口（portA/portB），且顺序敏感（$a < b \neq b < a$，$a \ll n \neq n \ll a$）。

sum 聚合无法区分 $(\mathbf{m}_A, \mathbf{m}_B)$ 和 $(\mathbf{m}_B, \mathbf{m}_A)$，必须显式分组：

$$\mathbf{m}_{\text{portA}} = \sum_{e: p_e = 0} \mathbf{m}_e, \quad \mathbf{m}_{\text{portB}} = \sum_{e: p_e = 1} \mathbf{m}_e$$

$$\mathbf{m}_{\text{pair}, t} = \text{MLP}_{\text{pair}}\left( [\mathbf{m}_{\text{portA}} \| \mathbf{m}_{\text{portB}} \| \mathbf{e}_t] \right)$$

其中 $\mathbf{e}_t = \text{Embedding}_{\text{type}}(t)$ 区分 ARITHMETIC/COMPARE/SHIFT 三种语义（$t \in \{0,1,2\}$），三者共享 MLP 权重，通过类型嵌入区分。

---

### 1.5 默认软混合（LOGIC/COMB 等）

对于不需要特化聚合的节点类型，使用可学习的 sum/mean/max 加权混合：

$$\mathbf{m}_{\text{soft}} = w_0 \cdot \mathbf{m}_{\text{sum}} + w_1 \cdot \mathbf{m}_{\text{mean}} + w_2 \cdot \mathbf{m}_{\text{max}}$$

其中 $[w_0, w_1, w_2] = \text{softmax}(\text{Embedding}_{\text{agg}}(\text{agg\_group}(t)))$，每个 agg_group 对应一组可学习权重。

---

### 1.6 路由链

路由采用 `torch.where` 实现无分支选择（内到外，高优先级覆盖低优先级）：

```
result = m_soft                              # 默认
result = where(is_shift, m_pair_shift, result)
result = where(is_compare, m_pair_cmp, result)
result = where(is_arith, m_pair_arith, result)
result = where(is_memory, m_mem, result)
result = where(is_sequential, m_seq, result)
result = where(has_ctrl_edge, m_mux, result)  # MUX 最高优先级
```

---

### 阶段 2：节点状态更新

$$\mathbf{h}_i^{(k)} = \text{LayerNorm}\left(\mathbf{h}_i^{(k-1)} + \text{DropPath}\left(\gamma^{(k)}\left(\mathbf{h}_i^{(k-1)}, \mathbf{m}_i^{(k)}\right)\right)\right)$$

### 阶段 3：边特征更新

$$\mathbf{e}_{j \to i}^{(k)} = \psi^{(k)} \left( \mathbf{h}_j^{(k)}, \mathbf{e}_{j \to i}^{(k-1)} \right)$$

- 边更新**仅依赖源节点**（驱动端），不依赖目标节点（负载端）
- **设计理由**：RTL 中导线（Wire）状态由驱动信号源决定，与目标节点（负载）无关

## 2. 与现有 GNN 变体的关系

| 模型 | 节点更新 | 边更新 | 特点 |
|------|---------|--------|------|
| GCN | $\sum_j \mathbf{h}_j$ | ❌ | 无边特征 |
| GAT | $\sum_j \alpha_{ij} \mathbf{h}_j$ | ❌ | 注意力权重 |
| GINEConv | $\sum_j (\mathbf{h}_j + \mathbf{e}_{ji})$ | ❌ | 边特征参与但不更新 |
| MPNN | $\sum_j \mathbf{W}_{e_{ji}} \mathbf{h}_j$ | ❌ | 边特征调制权重 |
| **本网络** | **类型感知差异化聚合** | ✅ $\psi(\mathbf{h}_j, \mathbf{e}_{ji})$ | **各节点类型电路语义 + 边仅依赖源节点** |

## 3. RTL 代码执行语义的模拟

### 3.1 节点 = 电路组件（Cell）

| RTL 概念 | NodeType | 特征含义 | 聚合策略 |
|----------|----------|----------|----------|
| MUX | `MUX`(6) | 选择状态、分支概率 | 控制门控 |
| DFF/寄存器 | `SEQUENTIAL`(4) | 时钟域、复位状态 | 嵌套门控残差 |
| SRAM/ROM | `MEMORY`(5) | 地址/数据/使能端口 | 端口角色缩放 |
| 加减乘除 | `ARITHMETIC`(7) | 操作数对 (A, B) | 端口对 MLP |
| 与或非 | `LOGIC`(8) | 置换不变逻辑 | 软混合 |
| 大小比较 | `COMPARE`(9) | 顺序敏感 (A, B) | 端口对 MLP |
| 移位器 | `SHIFT`(10) | 顺序敏感 (data, amt) | 端口对 MLP |
| 端口 | `INPUT`/`OUTPUT` | 驱动/负载特性 | 软混合 |

### 3.2 边 = 信号连接（Wire/Net）

| RTL 概念 | EdgeType | 索引 | 含义 |
|----------|----------|------|------|
| 数据线 | `DATA` | 0 | 通用数据流 |
| MUX 真分支 | `DATA_TRUE` | 1 | B 端口数据（S=1 时选择） |
| MUX 假分支 | `DATA_FALSE` | 2 | A 端口数据（S=0 时选择） |
| 控制信号 | `CONTROL` | 3 | 使能条件、分支选择 |
| 时钟 | `CLOCK` | 4 | 时钟域标识 |
| 复位 | `RESET` | 5 | 复位类型（同步/异步） |
| 使能 | `ENABLE` | 6 | 使能信号 |

### 3.3 消息传递 ≈ 信号传播

```
信号传播方向：
  Driver (node j) ──edge(j→i)──> Load (node i)

GNN 消息传递：
  h_j + e_{j→i} ──message──> type-aware aggregate at i ──update──> h_i'
```

## 4. 各节点类型聚合示例

### 4.1 MUX 节点示例

```verilog
assign out = sel ? in_true : in_false;
```

```
in_false (A) ──data_false──┐
                           ├──> MUX ──> out
in_true  (B) ──data_true───┘
                    ↑
        sel ────control────
```

```
m_ctrl  = φ_ctrl(h_sel, e_ctrl)
m_true  = φ(h_in_true,  e_data_true)
m_false = φ(h_in_false, e_data_false)
[α_true, α_false] = softmax(f_gate(m_ctrl))
m_gated = α_true · m_true + α_false · m_false
```

### 4.2 SEQUENTIAL 节点示例

```verilog
always_ff @(posedge clk or posedge rst) begin
    if (rst)       q <= 0;       // reset 最高优先级
    else if (en)   q <= d;       // enable 次之
    else           q <= q;       // 保持（clock 驱动）
end
```

```
g_r = σ(MLP(m_reset)) ≈ 1 when rst  → 屏蔽 enable/data/clock
g_e = σ(MLP(m_enable)) ≈ 1 when en  → 屏蔽 clock

m_seq = g_r · m_reset + (1-g_r) · (g_e · m_data + (1-g_e) · m_clock)
```

### 4.3 COMPARE/SHIFT 节点示例（顺序敏感）

```verilog
assign lt = a < b;     // a < b ≠ b < a，portA=a, portB=b
assign sh = a << n;    // a << n ≠ n << a
```

```
m_portA = φ(h_a, e_a)   [target_port_idx = 0]
m_portB = φ(h_b, e_b)   [target_port_idx = 1]
m_cmp = MLP_pair([m_portA || m_portB || e_compare])
# 交换 A/B 后结果不同，正确捕获顺序语义
```

## 5. 覆盖率预测的数学解释

本网络特别适合**边级覆盖率预测**任务：

$$\hat{y}_{j \to i} = \text{EdgeClassifier}\left(\mathbf{h}_{j}^{(K)} \| \mathbf{h}_{i}^{(K)} \| \mathbf{e}_{j \to i}^{(K)}\right)$$

经过 $K$ 层传播后，边特征 $\mathbf{e}_{j \to i}^{(K)}$ 编码了：

1. **局部结构**：源节点的类型和属性（通过边仅依赖源节点的更新规则）
2. **上下文信息**：$K$ 跳邻域内的拓扑和数据流模式
3. **ASM 融合信息**：通过 PerceiverCrossFusion 注入的测试激励上下文

## 6. 数学性质分析

### 6.1 表达能力

COMPARE/SHIFT 节点的端口对 MLP 使得聚合函数不再是置换不变的，可以区分 $(m_A, m_B)$ 和 $(m_B, m_A)$，超出标准 sum/mean/max 聚合的表达范围。

**定理**（非正式）：由于端口对 MLP 的顺序敏感性，本网络在 COMPARE/SHIFT 节点上可以区分某些 1-WL 测试无法区分的图结构。

### 6.2 参数效率

ARITHMETIC/COMPARE/SHIFT 共享同一个 `shared_pair_mlp`，通过类型嵌入 `pair_type_emb[t]` 区分语义。与三个独立 MLP 相比，参数量减少约 2/3，同时保留三类节点之间的共享表示能力。

### 6.3 过平滑问题

边更新机制有助于**缓解过平滑**：

- 传统 GNN：深层节点特征趋于相同
- 本网络：边特征保持了额外的区分信息，边特征的异质性可以"锚定"节点特征

## 7. 双图融合机制：ASM ↔ RTL PerceiverCrossFusion

本节描述 ASM（汇编测试激励）和 RTL（硬件设计）双图之间的特征融合机制。

### 7.1 设计理念

- **编码阶段**：ASM 和 RTL 通过双向 PerceiverCrossFusion 相互增强（ASM ↔ RTL）
- **预测阶段**：仅使用 RTL CDFG 进行边分类和图回归
- ASM CDFG 作为上下文信息，模拟"测试激励驱动硬件"

**信息流**：
```
ASM 图 ──GNN──> asm_h  ←──────────────────────────────────────
                  │                                             │
                  │ PerceiverCrossFusion(rtl_in, asm_in)       │
                  ▼                                             │
RTL 图 ──GNN──> rtl_h  ───────────────────────────────────────┘
                  │
                  ▼
            EdgeClassifier / GraphRegressor
```

### 7.2 PerceiverCrossFusion 的数学形式

每层 GNN 之后执行双向 Perceiver 融合。以 ASM→RTL 方向为例：

**步骤 1：Latent 初始化**

$$\mathbf{L} \in \mathbb{R}^{K \times d}, \quad K \ll N_{\text{asm}}$$

$K$ 个 latent token 作为信息瓶颈，$K=16$（默认）。

**步骤 2：Cross-Attention（latent 从 ASM 节点提炼）**

$$\mathbf{L}' = \text{CrossAttn}(\mathbf{L}, \mathbf{H}_{\text{asm}})$$

复杂度 $O(K \cdot N_{\text{asm}} \cdot d)$，$K \ll N_{\text{asm}}$ 时远低于全注意力 $O(N_{\text{asm}} \cdot N_{\text{rtl}} \cdot d)$。

**步骤 3：Cross-Attention（RTL 节点从 latent 读取）**

$$\mathbf{H}'_{\text{rtl}} = \text{CrossAttn}(\mathbf{H}_{\text{rtl}}, \mathbf{L}')$$

复杂度 $O(N_{\text{rtl}} \cdot K \cdot d)$。

**双向快照避免链式污染**：

```python
rtl_in, asm_in = rtl_h, asm_h  # 快照
rtl_h = fusion_asm2rtl(rtl_in, asm_in, ...)   # ASM → RTL
asm_h = fusion_rtl2asm(asm_in, rtl_in, ...)   # RTL → ASM（用旧 rtl_in）
```

若不用快照，`rtl_h` 已被 asm 融合调制，再用它去融合 asm 会引入链式污染。

### 7.3 PerceiverCrossFusion vs FiLM 对比

| 特性 | FiLM（旧方案） | PerceiverCrossFusion（当前） |
|------|--------------|------------------------------|
| 上下文粒度 | 全局池化（单向量） | K 个 latent token（细粒度） |
| 计算复杂度 | $O(N)$ 池化 + 广播 | $O((N_{\text{src}} + N_{\text{tgt}}) \cdot K \cdot d)$ |
| 注意力 | 无 | 节点级 cross-attention |
| 信息瓶颈 | 无（全部信息通过） | K tokens 强制选择性摘要 |
| 对应关系 | 无（全局调制） | 隐式 ASM 节点 ↔ RTL 节点对应 |

### 7.4 双图融合的物理意义

**为什么 Perceiver 适合 ASM → RTL 融合？**

1. **选择性注意力**：并非所有 ASM 指令都与每个 RTL 节点相关，latent token 学习哪些 ASM 指令对当前 RTL 上下文重要
2. **信息瓶颈**：$K$ 个 latent token 强制模型提炼出最关键的测试激励特征，而非直接广播全局均值
3. **节点级精度**：每个 RTL 节点独立从 latent 读取信息，细粒度远超 FiLM 的统一调制

## 8. 总结

| 设计决策 | 数学意义 | RTL 对应 |
|----------|----------|----------|
| MUX 控制门控 | $\alpha_{\text{true}} \cdot \mathbf{m}_{\text{true}} + \alpha_{\text{false}} \cdot \mathbf{m}_{\text{false}}$ | MUX 的互斥选择行为 |
| SEQUENTIAL 嵌套门控 | $g_r \odot m_r + (1-g_r) \odot (g_e \odot m_d + (1-g_e) \odot m_c)$ | reset > enable > data 严格优先级 |
| MEMORY 端口缩放 | $\sum_e \mathbf{s}_{p_e} \odot \mathbf{m}_e$ | 地址/数据/使能端口角色各异 |
| ARITH/CMP/SHIFT 端口对 | $\text{MLP}([\mathbf{m}_A \| \mathbf{m}_B \| \mathbf{e}_t])$ | 操作数顺序敏感（A op B ≠ B op A） |
| 软混合（LOGIC 等） | 可学习 sum/mean/max 加权 | 置换不变组合逻辑 |
| 路由链 `torch.where` | 无 GPU-CPU 同步的无分支选择 | 各节点类型并行计算，按类型选择 |
| 边更新仅依赖源节点 | $\psi(\mathbf{h}_j, \mathbf{e}_{ji})$ | 导线状态由驱动端决定 |
| PerceiverCrossFusion | 节点级 cross-attn，$K$ latent token 瓶颈 | 测试激励选择性激活硬件路径 |

本网络设计在数学上是**良定义的类型感知消息传递框架**，在物理上**精确模拟了 RTL 电路中各类型组件的信号传播语义**，是进行覆盖率预测的理想选择。

## 9. 整图回归任务

### 9.1 整图回归的信息流

| 任务类型 | 信息流向 | 输出 | 本网络适配度 |
|----------|----------|------|-------------|
| 边分类 | 局部 → 边 | $\hat{y}_{e} \in \{0,1\}$ | ✅ 天然适合 |
| 节点分类 | 邻域 → 节点 | $\hat{y}_{v} \in \mathbb{R}^C$ | ✅ 适合 |
| 整图回归 | 全图 → 标量 | $\hat{y}_{G} \in \mathbb{R}$ | ✅ RTL 图嵌入 |

### 9.2 RTL 覆盖率整图回归

整图回归目标：**预测整体覆盖率百分比**

$$\hat{y}_G = \text{MLP}_{\text{graph}}\left(\text{GlobalMeanPool}(\mathbf{H}_{\text{rtl}}^{(K)})\right)$$

**关键洞察**：整图覆盖率 ≈ 边级覆盖率的统计分布，而边级信息已通过节点特征（边更新仅依赖源节点）隐式编码进 RTL 节点嵌入。

### 9.3 多任务输出

```
RTL 节点特征 H_rtl [N, D]
       │
       ├── EdgeClassifier([h_src || h_tgt || e]) → edge_logits [E, 2]
       │
       └── GraphRegressor(GlobalMeanPool(H_rtl)) → coverage_rate [B, 1]
```

**ASM 特征仅在编码阶段使用**，预测阶段仅用 RTL 特征，这使得推理时无需汇编数据（可选）。

## 10. 参数规模

| 配置 | hidden_dim | GNN 层数 | 参数量（约） |
|------|------------|---------|-------------|
| Small | 128 | 2 | ~400K |
| Base | 256 | 4 | ~1.6M |

**差异化聚合新增参数（D=256，每 GNN 层）**：

| 模块 | 参数数 |
|------|--------|
| `seq_gate_reset` + `seq_gate_enable` | 2 × (256×256 + 256) = 131K |
| `pair_type_emb` + `shared_pair_mlp` | 3×256 + (768×512 + 512×256) = 527K |
| `mem_port_scale_emb` | 8×256 = 2K |
| **单层合计** | **~660K** |
| **4 层总增量** | **~2.64M** |

## 11. 参考文献

- **Perceiver IO**: Jaegle et al., "Perceiver IO: A General Architecture for Structured Inputs & Outputs" (ICML 2022)
- **GIN**: Xu et al., "How Powerful are Graph Neural Networks?" (ICLR 2019)
- **DropPath**: Huang et al., "Deep Networks with Stochastic Depth" (ECCV 2016)
- **Label Smoothing**: Szegedy et al., "Rethinking the Inception Architecture" (CVPR 2016)
- **DeepSets**: Zaheer et al., "Deep Sets" (NeurIPS 2017)
