# 双图特征融合方法研究

## 背景

**任务目标**：模拟"测试激励（ASM）在 RTL 硬件上执行"的语义

**当前方案**：双向 FiLM 注入
- ASM → RTL：测试激励驱动硬件状态
- RTL → ASM：硬件状态反馈给激励

**数据特性**：ASM CDFG 仅有图结构，没有指令执行顺序信息

---

## 方法一：Cross-Graph Attention（跨图注意力）

### 来源
- Graph Matching Networks (Li et al., ICML 2019)
- Cross-Modal Attention (多模态学习领域)

### 核心思想
让 RTL 节点"关注" ASM 节点，学习细粒度的节点级对应关系。

### 公式
```
# RTL 节点关注 ASM 节点
Q_rtl = W_q · h_rtl           # [N_rtl, D]
K_asm = W_k · h_asm           # [N_asm, D]
V_asm = W_v · h_asm           # [N_asm, D]

Attention = softmax(Q_rtl · K_asm^T / √D)  # [N_rtl, N_asm]
h_rtl' = h_rtl + Attention · V_asm         # [N_rtl, D]
```

### 语义解释
- **物理含义**：每个 RTL 节点（硬件单元）学习它与哪些 ASM 指令相关
- **例如**：ALU 节点可能关注算术指令，寄存器节点关注 load/store 指令

### 优缺点
| 优点 | 缺点 |
|------|------|
| 细粒度节点级交互 | 计算复杂度 O(N_rtl × N_asm) |
| 可解释性强（注意力可视化） | 需要 dense batch 转换 |
| 能学习显式对应关系 | 参数量较大 |

### 适用场景
- 需要理解"哪条指令影响哪个硬件单元"
- 图规模较小（< 1000 节点）

---

## 方法二：Graph Coarsening + Hierarchical Fusion（图粗化 + 层次融合）

### 来源
- DiffPool (Ying et al., NeurIPS 2018)
- Graph U-Net (Gao & Ji, ICML 2019)

### 核心思想
将两个图分别粗化到相同粒度，然后在粗化后的"超节点"级别进行融合。

### 架构
```
ASM Graph                          RTL Graph
    │                                  │
    ▼ (GNN)                            ▼ (GNN)
ASM Nodes [N_asm, D]              RTL Nodes [N_rtl, D]
    │                                  │
    ▼ (DiffPool)                       ▼ (DiffPool)
ASM Clusters [K, D]               RTL Clusters [K, D]
    │                                  │
    └──────────┬───────────────────────┘
               ▼
        Cluster-level Fusion
        (Concat / Attention / FiLM)
               │
               ▼
        Fused Clusters [K, D]
               │
               ▼ (Unpool)
        Back to RTL Nodes [N_rtl, D]
```

### 语义解释
- **物理含义**：将 ASM 指令聚类为"功能块"，RTL 节点聚类为"硬件模块"
- **融合语义**：功能块与硬件模块的对应关系

### 优缺点
| 优点 | 缺点 |
|------|------|
| 计算复杂度降低 O(K²) | 需要学习聚类分配 |
| 层次化语义更清晰 | 实现复杂度高 |
| 可处理大规模图 | 信息损失风险 |

### 适用场景
- 大规模图（> 10000 节点）
- 需要层次化理解

---

## 方法三：Virtual Node（虚拟节点消息传递）

### 来源
- OGB-LSC (Hu et al., NeurIPS 2021)
- Graphormer (Ying et al., NeurIPS 2021)

### 核心思想
添加"虚拟节点"作为两个图之间的信息桥梁。

### 架构
```
┌─────────────────────────────────────────────────────────┐
│                    Virtual Node (v)                      │
│                         ↑↓                               │
│         ┌───────────────┴───────────────┐               │
│         ↓                               ↓               │
│    ASM Graph                       RTL Graph            │
│    (all nodes ↔ v)                (all nodes ↔ v)       │
└─────────────────────────────────────────────────────────┘

每层 GNN:
1. ASM 节点 → 虚拟节点: v = v + mean(h_asm)
2. RTL 节点 → 虚拟节点: v = v + mean(h_rtl)
3. 虚拟节点 → ASM 节点: h_asm = h_asm + MLP(v)
4. 虚拟节点 → RTL 节点: h_rtl = h_rtl + MLP(v)
```

### 语义解释
- **物理含义**：虚拟节点代表"执行上下文"或"系统状态"
- **信息流**：ASM 指令通过虚拟节点影响 RTL，反之亦然

### 优缺点
| 优点 | 缺点 |
|------|------|
| 实现简单 | 信息瓶颈（单节点） |
| 计算高效 O(N_asm + N_rtl) | 缺乏细粒度交互 |
| 与现有 GNN 兼容 | 可解释性较弱 |

### 适用场景
- 需要全局信息传递
- 计算资源受限

---

## 方法四：Contrastive Learning（对比学习）

### 来源
- SimCLR (Chen et al., ICML 2020)
- GraphCL (You et al., NeurIPS 2020)

### 核心思想
通过对比学习让匹配的 (ASM, RTL) 对在嵌入空间中接近。

### 架构
```
# 编码
z_asm = ASM_Encoder(asm_graph)  # [B, D]
z_rtl = RTL_Encoder(rtl_graph)  # [B, D]

# 对比损失（InfoNCE）
# 正样本：同一对 (asm_i, rtl_i)
# 负样本：不同对 (asm_i, rtl_j) where i ≠ j

L_contrast = -log(exp(sim(z_asm_i, z_rtl_i)/τ) /
                  Σ_j exp(sim(z_asm_i, z_rtl_j)/τ))

# 总损失
L_total = L_task + λ · L_contrast
```

### 语义解释
- **物理含义**：学习 ASM 和 RTL 的语义对齐
- **辅助任务**：帮助模型理解"哪个测试激励对应哪个硬件行为"

### 优缺点
| 优点 | 缺点 |
|------|------|
| 无需显式融合 | 需要足够的负样本 |
| 学习语义对齐 | 对 batch size 敏感 |
| 可作为辅助损失 | 可能与主任务冲突 |

### 适用场景
- 作为辅助训练目标
- 数据量充足

---

## 方法五：Hypergraph Neural Network（超图神经网络）

### 来源
- HGNN (Feng et al., AAAI 2019)
- HyperGCN (Yadati et al., NeurIPS 2019)

### 核心思想
将 ASM 和 RTL 节点放入同一个超图，用超边连接相关节点。

### 架构
```
超图构建:
- 节点集: V = V_asm ∪ V_rtl
- 超边集: E = {e_1, e_2, ...}
  - 每条超边连接一组相关的 ASM 和 RTL 节点

超图卷积:
H = D_v^{-1/2} · H · W · D_e^{-1} · H^T · D_v^{-1/2} · X · Θ

其中:
- H: 关联矩阵 [|V|, |E|]
- D_v, D_e: 度矩阵
```

### 语义解释
- **物理含义**：超边表示"一组指令影响一组硬件单元"
- **高阶关系**：捕获多对多的复杂关系

### 优缺点
| 优点 | 缺点 |
|------|------|
| 建模高阶关系 | 超边构建需要先验知识 |
| 统一的图表示 | 实现复杂 |
| 灵活的连接模式 | 计算开销较大 |

### 适用场景
- 存在复杂的多对多关系
- 有先验知识构建超边

---

## 方法六：Neural Execution Engine（神经执行引擎）

### 来源
- Neural Turing Machine (Graves et al., 2014)
- Neural Program Interpreter (Reed & de Freitas, ICLR 2016)

### 核心思想
设计一个可微分的"执行器"，模拟指令在硬件上的执行。

### 架构
```
# 初始化硬件状态
state = RTL_Encoder(rtl_graph)  # [N_rtl, D]

# 逐指令执行
for instr in asm_instructions:
    # 解码指令
    op, operands = Instruction_Decoder(instr)

    # 选择受影响的硬件单元（软注意力）
    attention = Attention(state, instr)  # [N_rtl]

    # 执行操作（可微分）
    update = Execute(op, operands, state)  # [N_rtl, D]

    # 更新状态
    state = state + attention.unsqueeze(-1) * update

# 输出最终状态
output = state
```

### 语义解释
- **物理含义**：最接近真实硬件执行的语义
- **可解释性**：可以追踪每条指令的影响

### 优缺点
| 优点 | 缺点 |
|------|------|
| 语义最精确 | 实现复杂度最高 |
| 高度可解释 | 需要指令语义知识 |
| 可建模精确执行 | 训练困难 |

### 适用场景
- 需要精确建模执行语义
- 有充足的领域知识

---

## 方法对比总结

| 方法 | 复杂度 | 语义精度 | 实现难度 | 推荐场景 |
|------|--------|----------|----------|----------|
| **FiLM（当前）** | O(N) | 中 | 低 | 通用基线 |
| Cross-Graph Attention | O(N²) | 高 | 中 | 小图、需要可解释性 |
| Graph Coarsening | O(K²) | 中 | 高 | 大规模图 |
| Virtual Node | O(N) | 低 | 低 | 快速实验 |
| Contrastive Learning | O(B²) | 中 | 低 | 辅助损失 |
| Hypergraph | O(E×N) | 高 | 高 | 多对多关系 |
| Neural Execution | O(T×N) | 最高 | 最高 | 精确建模 |

---

## 推荐方案

### 方案 A：FiLM + Contrastive（低风险增强）

在当前双向 FiLM 基础上，添加对比学习作为辅助损失：

```python
L_total = L_edge + L_graph + λ · L_contrast
```

**优点**：改动小，可能提升表示质量
**风险**：低

### 方案 B：Cross-Graph Attention（中等改动）

替换 FiLM 为跨图注意力，获得细粒度交互：

```python
# 每层 GNN 后
rtl_h = rtl_h + CrossAttention(rtl_h, asm_h)
```

**优点**：可解释性强，细粒度建模
**风险**：中（计算开销增加）

### 方案 C：Virtual Node（简化方案）

使用虚拟节点替代 FiLM，更简单的全局信息传递：

```python
# 每层 GNN 后
v = v + mean(h_asm) + mean(h_rtl)
h_asm = h_asm + MLP_asm(v)
h_rtl = h_rtl + MLP_rtl(v)
```

**优点**：实现简单，计算高效
**风险**：低（但表达能力可能不如 FiLM）

---

## 参考文献

- **FiLM**: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer" (AAAI 2018)
- **Graph Matching Networks**: Li et al., "Graph Matching Networks for Learning the Similarity of Graph Structured Objects" (ICML 2019)
- **DiffPool**: Ying et al., "Hierarchical Graph Representation Learning with Differentiable Pooling" (NeurIPS 2018)
- **Virtual Node**: Hu et al., "OGB-LSC: A Large-Scale Challenge for Machine Learning on Graphs" (NeurIPS 2021)
- **SimCLR**: Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations" (ICML 2020)
- **GraphCL**: You et al., "Graph Contrastive Learning with Augmentations" (NeurIPS 2020)
- **HGNN**: Feng et al., "Hypergraph Neural Networks" (AAAI 2019)
- **Neural Turing Machine**: Graves et al., "Neural Turing Machines" (arXiv 2014)
- **Neural Program Interpreter**: Reed & de Freitas, "Neural Programmer-Interpreters" (ICLR 2016)

---

# 2024+ 新方法

以下是 2024 年之后提出的双图/多模态图融合新方法：

---

## 方法七：Graph Mamba / State Space Models for Graphs

### 来源
- Graph Mamba (Wang et al., 2024)
- Graph-S4 系列

### 核心思想
将 Mamba (State Space Model) 应用于图结构，用选择性状态空间替代注意力机制。

### 架构
```
# 交替状态空间扫描
for layer in layers:
    asm_h = GraphMamba(asm_h, asm_edges)
    rtl_h = GraphMamba(rtl_h, rtl_edges)
    # 跨图状态注入
    rtl_h = rtl_h + SelectiveSSM(asm_context)
```

### 语义解释
- **物理含义**：ASM 状态空间驱动 RTL 状态演化
- **选择性机制**：模型学习"哪些 ASM 信息对当前 RTL 节点重要"

### 优缺点
| 优点 | 缺点 |
|------|------|
| 线性复杂度 O(N) | 需要定义图上的扫描顺序 |
| 长程依赖建模能力强 | 相对较新，生态不完善 |
| 适合大规模图 | 实现复杂度中等 |

### 适用场景
- 大规模图（> 10000 节点）
- 需要建模长程依赖
- 计算资源受限

---

## 方法八：Graph Mixture of Experts (Graph MoE)

### 来源
- 受 Mixtral/Switch Transformer 启发的图版本 (2024)

### 核心思想
不同的专家网络处理不同类型的跨图交互，通过路由器动态选择。

### 架构
```
# 路由器决定使用哪个专家
router_logits = Router(asm_context, rtl_context)
expert_weights = TopK_Softmax(router_logits, k=2)

# 稀疏专家融合
output = Σ expert_weights[i] * Expert_i(asm_h, rtl_h)
```

### 语义解释
- **物理含义**：不同专家学习不同的"执行模式"
- **例如**：专家 1 处理算术运算，专家 2 处理内存访问，专家 3 处理控制流

### 优缺点
| 优点 | 缺点 |
|------|------|
| 稀疏激活，计算高效 | 路由器训练不稳定 |
| 不同专家学习不同模式 | 负载均衡问题 |
| 可扩展性强 | 实现复杂度中等 |

### 适用场景
- 存在多种不同的交互模式
- 需要可扩展的模型容量
- 计算资源充足

---

## 方法九：Graph Flow Matching

### 来源
- Flow Matching for Graphs (2024)
- 受 Flow Matching 生成模型启发

### 核心思想
学习从 ASM 图到 RTL 图的连续流变换，通过 ODE 求解器进行融合。

### 架构
```
# 学习 ASM → RTL 的流场
v_t = FlowNetwork(asm_h, rtl_h, t)

# 通过 ODE 求解器积分
rtl_h_fused = rtl_h + ∫₀¹ v_t dt
```

### 语义解释
- **物理含义**：ASM 特征"流动"到 RTL 特征空间
- **连续变换**：建模从"测试激励"到"硬件状态"的连续演化

### 优缺点
| 优点 | 缺点 |
|------|------|
| 可建模复杂的图到图映射 | 计算复杂度 O(N×T) |
| 生成式视角，可用于数据增强 | 实现复杂度高 |
| 理论基础扎实 | 训练不稳定 |

### 适用场景
- 需要建模复杂的映射关系
- 数据增强需求
- 研究导向的项目

---

## 方法十：Geometric Algebra Graph Networks

### 来源
- Clifford Group Equivariant Networks (2024)

### 核心思想
使用几何代数（Clifford 代数）表示图特征，保持几何等变性。

### 架构
```
# 几何代数特征
asm_mv = ToMultivector(asm_h)  # 多向量表示
rtl_mv = ToMultivector(rtl_h)

# 几何积融合
fused_mv = GeometricProduct(asm_mv, rtl_mv)
rtl_h_fused = FromMultivector(fused_mv)
```

### 语义解释
- **物理含义**：几何积捕获两个图之间的"几何关系"
- **多向量表示**：标量（全局信息）+ 向量（方向信息）+ 双向量（旋转信息）

### 优缺点
| 优点 | 缺点 |
|------|------|
| 更丰富的特征表示 | 实现复杂度高 |
| 天然支持多模态融合 | 需要几何代数知识 |
| 物理可解释性强 | 计算开销较大 |

### 适用场景
- 需要物理可解释性
- 特征具有几何意义
- 研究导向的项目

---

## 方法十一：Latent Graph Diffusion

### 来源
- 受 Latent Diffusion Models 启发 (2024)

### 核心思想
在潜在空间中进行图扩散，融合两个图的信息。

### 架构
```
# 编码到潜在空间
z_asm = Encoder(asm_graph)
z_rtl = Encoder(rtl_graph)

# 潜在空间扩散融合
z_fused = DiffusionProcess(z_asm, z_rtl, steps=T)

# 解码回图空间
rtl_h_fused = Decoder(z_fused)
```

### 语义解释
- **物理含义**：在抽象的"语义空间"中融合 ASM 和 RTL
- **扩散过程**：逐步将 ASM 信息"扩散"到 RTL 表示中

### 优缺点
| 优点 | 缺点 |
|------|------|
| 强大的表示学习能力 | 计算复杂度 O(N×T) |
| 可处理噪声和不完整数据 | 训练时间长 |
| 生成式建模能力 | 实现复杂度高 |

### 适用场景
- 数据有噪声或不完整
- 需要生成式建模
- 计算资源充足

---

## 方法十二：Relational Bottleneck

### 来源
- Relational Bottleneck for Graph Neural Networks (ICML 2024 类似工作)

### 核心思想
通过信息瓶颈学习两个图之间的关键关系，压缩冗余信息。

### 架构
```
# 学习关系瓶颈
R = RelationEncoder(asm_h, rtl_h)  # [K, D] K个关系
R_compressed = InformationBottleneck(R)

# 通过瓶颈融合
rtl_h_fused = rtl_h + RelationDecoder(R_compressed, rtl_h)
```

### 语义解释
- **物理含义**：发现"哪些指令模式影响哪些硬件模块"
- **信息瓶颈**：只保留对任务有用的关系信息

### 优缺点
| 优点 | 缺点 |
|------|------|
| 自动发现重要的跨图关系 | 瓶颈大小需要调参 |
| 信息压缩，防止过拟合 | 可能丢失有用信息 |
| 可解释性强 | 实现复杂度中等 |

### 适用场景
- 需要发现关键关系
- 数据量有限，需要防止过拟合
- 需要可解释性

---

## 方法十三：Tokenized Graph Transformer

### 来源
- 受 Vision Transformer 的 patch tokenization 启发 (2024)

### 核心思想
将图分割为 token，使用标准 Transformer 处理，实现跨图交互。

### 架构
```
# 图 tokenization
asm_tokens = GraphTokenizer(asm_graph)  # [T_asm, D]
rtl_tokens = GraphTokenizer(rtl_graph)  # [T_rtl, D]

# 拼接后用 Transformer 处理
all_tokens = [CLS, asm_tokens, SEP, rtl_tokens]
fused_tokens = Transformer(all_tokens)

# 解码回图
rtl_h_fused = TokenToGraph(fused_tokens[T_asm+2:])
```

### 语义解释
- **物理含义**：将 ASM 和 RTL 视为两个"文档"，用 Transformer 理解它们的关系
- **统一建模**：跨图交互通过 self-attention 自然实现

### 优缺点
| 优点 | 缺点 |
|------|------|
| 可直接使用预训练 Transformer | 计算复杂度 O(T²) |
| 统一的序列建模 | 图结构信息可能丢失 |
| 灵活的跨图交互 | tokenization 策略影响大 |

### 适用场景
- 想利用预训练模型
- 图规模较小
- 快速实验

---

## 2024+ 方法对比总结

| 方法 | 复杂度 | 创新点 | 实现难度 | 适用场景 |
|------|--------|--------|----------|----------|
| Graph Mamba | O(N) | SSM 替代注意力 | 中 | 大规模图、长程依赖 |
| Graph MoE | O(N×k) | 稀疏专家 | 中 | 多样化交互模式 |
| Flow Matching | O(N×T) | 连续流变换 | 高 | 复杂映射、数据增强 |
| Geometric Algebra | O(N) | 几何等变性 | 高 | 物理语义建模 |
| Latent Diffusion | O(N×T) | 潜在空间扩散 | 高 | 噪声鲁棒、生成 |
| Relational Bottleneck | O(N×K) | 信息瓶颈 | 中 | 关系发现、可解释 |
| Tokenized Transformer | O(T²) | 统一序列建模 | 低 | 预训练迁移 |

---

## 2024+ 推荐方案

### 首选：Graph Mamba + FiLM

结合 Mamba 的高效长程建模和 FiLM 的条件注入：
- **复杂度**：O(N)
- **实现难度**：中
- **语义**：ASM 状态空间驱动 RTL 状态演化

### 备选：Relational Bottleneck

学习 ASM-RTL 之间的关键关系：
- **复杂度**：O(N×K)
- **实现难度**：中
- **语义**：发现"哪些指令模式影响哪些硬件模块"

### 简单方案：Tokenized Transformer

如果想快速实验：
- **复杂度**：O(T²)
- **实现难度**：低
- **语义**：统一序列建模，可用预训练权重

---

# SSM-FiLM 融合方案详细设计

## 设计约束

- **保持不变**：`GNNLayer`、`ControlGatedGNNLayer` 消息传递逻辑
- **仅修改**：特征融合模块（当前为 `FiLMInjection`）
- **目标**：用 SSM 的选择性状态空间机制增强 FiLM 的条件注入

---

## 核心思想

将 Mamba 的**选择性状态空间机制**应用于跨图特征融合，而非图内消息传递。

**当前 FiLM 融合**：
```
context = global_mean_pool(asm_h)  # [B, D] 静态上下文
rtl_h' = (1 + γ(context)) ⊙ rtl_h + β(context)  # 全局调制
```

**SSM-FiLM 融合**：
```
# 1. 将 ASM 节点序列化（按拓扑序或 BFS）
asm_seq = topological_sort(asm_h)  # [B, T_asm, D]

# 2. SSM 处理：学习"执行状态"的演化
state = SSM(asm_seq)  # [B, T_asm, D] 每个位置的状态

# 3. 选择性聚合：根据 RTL 节点选择相关状态
context = SelectivePool(state, rtl_h)  # [B, D] 或 [N_rtl, D]

# 4. FiLM 注入
rtl_h' = (1 + γ(context)) ⊙ rtl_h + β(context)
```

---

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│ SSMFiLMInjection (替代 FiLMInjection)                       │
│                                                             │
│   ASM 节点特征 [N_asm, D]                                   │
│         │                                                   │
│         ▼ (按 batch 分组 + 拓扑排序)                        │
│   ASM 序列 [B, T_max, D] (padding)                          │
│         │                                                   │
│         ▼ (Mamba Block)                                     │
│   ┌─────────────────────────────────────────┐              │
│   │ x → Linear → Conv1d → SSM → Linear → x' │              │
│   │              ↑                           │              │
│   │         Δ, B, C (选择性参数)             │              │
│   └─────────────────────────────────────────┘              │
│         │                                                   │
│         ▼ (选择性聚合)                                      │
│   Context [B, D] 或 [N_rtl, D]                              │
│         │                                                   │
│         ▼ (FiLM 注入)                                       │
│   RTL 节点特征' = (1 + γ) ⊙ RTL + β                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## SSM 核心公式

Mamba 的选择性状态空间模型：

```
# 离散化状态空间
h_t = Ā · h_{t-1} + B̄ · x_t    # 状态更新
y_t = C · h_t                   # 输出

其中：
- Ā = exp(Δ · A)               # 离散化状态矩阵
- B̄ = (Δ · A)^{-1} · (Ā - I) · Δ · B
- Δ, B, C 是输入依赖的（选择性）
```

**选择性机制的语义**：
- **Δ（步长）**：控制"记忆"多少历史信息
- **B（输入门）**：控制当前输入的影响
- **C（输出门）**：控制输出哪些状态信息

---

## 选择性聚合策略

三种聚合方式（可配置）：

### 1. 全局聚合（简单）

```python
context = state[:, -1, :]  # 取最后状态 [B, D]
# 或
context = state.mean(dim=1)  # 平均池化 [B, D]
```

### 2. 注意力聚合（中等）

```python
# RTL 图级特征作为 query
rtl_query = global_mean_pool(rtl_h, rtl_batch)  # [B, D]
attn = softmax(rtl_query @ state.transpose(-1, -2))  # [B, 1, T]
context = (attn @ state).squeeze(1)  # [B, D]
```

### 3. 节点级聚合（精细）

```python
# 每个 RTL 节点独立查询
attn = softmax(rtl_h @ state[rtl_batch].transpose(-1, -2))  # [N_rtl, 1, T]
context = (attn @ state[rtl_batch]).squeeze(1)  # [N_rtl, D]
```

---

## 简化版 SSM 实现

无需 CUDA 内核的纯 PyTorch 实现：

```python
class SimpleSSM(nn.Module):
    """简化版选择性状态空间模型"""

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_state = d_state

        # 状态空间参数（可学习）
        self.A = nn.Parameter(torch.randn(d_model, d_state))

        # 选择性参数生成器（输入依赖）
        self.delta_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, T, D] 输入序列
        Returns:
            y: [B, T, D] 输出序列
        """
        B, T, D = x.shape

        # 选择性参数（输入依赖）
        delta = F.softplus(self.delta_proj(x))  # [B, T, D]
        B_t = self.B_proj(x)  # [B, T, d_state]
        C_t = self.C_proj(x)  # [B, T, d_state]

        # 离散化
        A_bar = torch.exp(delta.unsqueeze(-1) * self.A)  # [B, T, D, d_state]

        # 状态递推
        h = x.new_zeros(B, D, self.d_state)
        outputs = []
        for t in range(T):
            h = A_bar[:, t] * h + delta[:, t, :, None] * B_t[:, t, None, :]
            y_t = (h * C_t[:, t, None, :]).sum(-1)  # [B, D]
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # [B, T, D]
```

---

## SSMFiLMInjection 完整实现

```python
class SSMFiLMInjection(nn.Module):
    """
    SSM 增强的 FiLM 注入模块

    用 SSM 处理源图节点序列，生成动态上下文，然后进行 FiLM 注入
    """

    def __init__(
        self,
        hidden_dim: int,
        d_state: int = 16,
        pool_mode: str = "last",  # "last", "mean", "attention"
        init_std: float = 0.02,
    ):
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
        target_h: Tensor,          # [N_tgt, D] 目标节点特征
        source_h: Tensor,          # [N_src, D] 源节点特征
        target_batch: Tensor,      # [N_tgt] 目标 batch 索引
        source_batch: Tensor,      # [N_src] 源 batch 索引
    ) -> Tensor:
        """
        Args:
            target_h: 目标图节点特征（如 RTL）
            source_h: 源图节点特征（如 ASM）
            target_batch: 目标节点的 batch 索引
            source_batch: 源节点的 batch 索引

        Returns:
            调制后的目标节点特征
        """
        # 1. 将源节点按 batch 分组并 padding 成序列
        source_seq, seq_mask = self._batch_to_sequence(source_h, source_batch)
        # source_seq: [B, T_max, D], seq_mask: [B, T_max]

        # 2. SSM 处理
        ssm_out = self.ssm(source_seq)  # [B, T_max, D]

        # 3. 选择性聚合
        if self.pool_mode == "last":
            # 取每个序列的最后一个有效位置
            lengths = seq_mask.sum(dim=1).long() - 1  # [B]
            context = ssm_out[torch.arange(ssm_out.size(0)), lengths]  # [B, D]
        elif self.pool_mode == "mean":
            # 平均池化（mask 掉 padding）
            ssm_out = ssm_out * seq_mask.unsqueeze(-1)
            context = ssm_out.sum(dim=1) / seq_mask.sum(dim=1, keepdim=True)  # [B, D]
        elif self.pool_mode == "attention":
            # 注意力聚合
            target_graph = global_mean_pool(target_h, target_batch)  # [B, D]
            query = self.query_proj(target_graph).unsqueeze(1)  # [B, 1, D]
            key = self.key_proj(ssm_out)  # [B, T, D]
            attn = torch.bmm(query, key.transpose(1, 2)) / (self.hidden_dim ** 0.5)
            attn = attn.masked_fill(~seq_mask.unsqueeze(1), float('-inf'))
            attn = F.softmax(attn, dim=-1)  # [B, 1, T]
            context = torch.bmm(attn, ssm_out).squeeze(1)  # [B, D]

        # 4. FiLM 注入
        context_expanded = context[target_batch]  # [N_tgt, D]
        gamma = self.gamma_gen(context_expanded)
        beta = self.beta_gen(context_expanded)

        return (1 + gamma) * target_h + beta

    def _batch_to_sequence(
        self, h: Tensor, batch: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """将 batch 格式转换为 padding 序列格式"""
        device = h.device
        batch_size = batch.max().item() + 1

        # 计算每个 batch 的节点数
        counts = torch.bincount(batch, minlength=batch_size)
        max_len = counts.max().item()

        # 初始化
        seq = h.new_zeros(batch_size, max_len, h.size(-1))
        mask = h.new_zeros(batch_size, max_len, dtype=torch.bool)

        # 填充
        for b in range(batch_size):
            idx = (batch == b).nonzero(as_tuple=True)[0]
            length = len(idx)
            seq[b, :length] = h[idx]
            mask[b, :length] = True

        return seq, mask
```

---

## 与当前架构的集成

```python
class FiLMDualEncoder(nn.Module):
    def __init__(
        self,
        ...,
        fusion_type: str = "film",  # "film" 或 "ssm_film"
        ssm_pool_mode: str = "last",
    ):
        ...
        # 根据配置选择融合模块
        if fusion_type == "film":
            self.film_asm2rtl = nn.ModuleList(
                [FiLMInjection(hidden_dim) for _ in range(num_layers)]
            )
            self.film_rtl2asm = nn.ModuleList(
                [FiLMInjection(hidden_dim) for _ in range(num_layers)]
            )
        elif fusion_type == "ssm_film":
            self.film_asm2rtl = nn.ModuleList(
                [SSMFiLMInjection(hidden_dim, pool_mode=ssm_pool_mode)
                 for _ in range(num_layers)]
            )
            self.film_rtl2asm = nn.ModuleList(
                [SSMFiLMInjection(hidden_dim, pool_mode=ssm_pool_mode)
                 for _ in range(num_layers)]
            )
```

---

## 语义解释

**为什么 SSM 适合 ASM → RTL 融合？**

| 特性 | 语义解释 |
|------|----------|
| **状态递推** | 模拟"指令逐条执行"的状态变化 |
| **选择性 Δ** | 学习"哪些指令对当前 RTL 节点重要" |
| **长程依赖** | 捕获复杂测试激励中的远距离依赖 |
| **线性复杂度** | O(T) 替代注意力的 O(T²) |

---

## FiLM vs SSM-FiLM 对比

| 特性 | FiLM | SSM-FiLM |
|------|------|----------|
| 上下文 | 静态（全局池化） | 动态（状态演化） |
| 复杂度 | O(N) | O(N + T) |
| 长程依赖 | 弱（单次池化） | 强（状态递推） |
| 选择性 | 无 | 有（输入依赖参数） |
| 实现难度 | 低 | 中 |
| 参数量 | 2D² | 2D² + 3D×d_state |

---

## 2024+ 参考文献

- **Graph Mamba**: Wang et al., "Graph Mamba: Towards Learning on Graphs with State Space Models" (2024)
- **Mamba**: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2024)
- **Mixtral**: Jiang et al., "Mixtral of Experts" (2024)
- **Flow Matching**: Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
- **Clifford GNN**: Ruhe et al., "Clifford Group Equivariant Neural Networks" (NeurIPS 2023)
- **Latent Diffusion**: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (CVPR 2022)
- **Information Bottleneck**: Tishby et al., "The Information Bottleneck Method" (1999)
