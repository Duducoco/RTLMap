# RTLMap: Mapping RTL States to Hyperrectangles for Pre-Simulation Stimulus Prioritization in Processor Verification

## RSA-GNN

VeriPCP 将处理器 RTL 转换为控制数据流图（Control/Data-Flow Graph, CDFG），并使用控制感知图神经网络编码其结构与语义。给定 RTL 图

$$
\mathcal{G}_{\mathrm{RTL}}=(\mathcal{V},\mathcal{E}),
$$

其中节点 $v_i\in\mathcal{V}$ 表示 RTL cell，有向边 $e_{ji}=(v_j,v_i)\in\mathcal{E}$ 表示从源节点 $v_j$ 到目标节点 $v_i$ 的信号依赖。编码器输出节点表示、边表示和图级 RTL 表示，用于后续覆盖率预测与 stimulus prioritization。

### 1. Input Feature Encoding

节点 $v_i$ 包含 cell 类型 $c_i$ 和信号位宽 $w_i$。其初始表示为

$$
\mathbf{h}_i^{(0)}=
\operatorname{LN}\left(
\operatorname{Emb}_{\mathrm{cell}}(c_i)
+\mathbf{W}_{w}\log_2(w_i+1)+\mathbf{b}_{w}
\right),
$$

其中 $\operatorname{Emb}_{\mathrm{cell}}$ 是可学习的 cell-type embedding，$\operatorname{LN}$ 表示 Layer Normalization。对位宽取对数可以压缩 RTL 中跨度较大的数值范围。

每条边 $e_{ji}$ 包含边类型 $r_{ji}$、信号位宽 $w_{ji}$、源端口位置 $p_{ji}^{s}$ 和目标端口位置 $p_{ji}^{t}$。边类型包括 `DATA`、`DATA_TRUE`、`DATA_FALSE`、`CONTROL`、`CLOCK`、`RESET` 和 `ENABLE`。边的初始表示为

$$
\mathbf{e}_{ji}^{(0)}=
\operatorname{Emb}_{\mathrm{edge}}(r_{ji})
+\mathbf{W}_{e}\log_2(w_{ji}+1)+\mathbf{b}_{e}
+\left[
\operatorname{Emb}_{s}(p_{ji}^{s})\,\Vert\,
\operatorname{Emb}_{t}(p_{ji}^{t})
\right],
$$

其中 $\Vert$ 表示向量拼接。端口位置使模型能够区分不同的操作数角色，例如减法器的输入 A 与输入 B。

### 2. Edge Message Construction

第 $\ell$ 层将源节点状态与边状态拼接。控制边和其他边分别使用独立的消息函数：

$$
\mathbf{m}_{ji}^{(\ell)}=
\begin{cases}
\phi_{\mathrm{ctrl}}^{(\ell)}
\left([\mathbf{h}_{j}^{(\ell)}\Vert\mathbf{e}_{ji}^{(\ell)}]\right),
& r_{ji}=\mathrm{CONTROL},\\[4pt]
\phi_{\mathrm{data}}^{(\ell)}
\left([\mathbf{h}_{j}^{(\ell)}\Vert\mathbf{e}_{ji}^{(\ell)}]\right),
& \text{otherwise},
\end{cases}
$$

其中 $\phi_{\mathrm{ctrl}}$ 和 $\phi_{\mathrm{data}}$ 均为带 GELU 激活函数的两层 MLP。编码器还为每条入边计算注意力权重：

$$
a_{ji}^{(\ell)}=
\mathbf{w}_{a}^{\top}
[\mathbf{h}_{j}^{(\ell)}\Vert\mathbf{e}_{ji}^{(\ell)}]+b_a,
$$

$$
\alpha_{ji}^{(\ell)}=
\frac{\exp(a_{ji}^{(\ell)})}
{\sum_{k\in\mathcal{N}(i)}\exp(a_{ki}^{(\ell)})}.
$$

### 3. Control-Aware Aggregation

RTL 节点具有不同的电路语义，因此 VeriPCP 根据目标节点类型选择聚合路径，而不是对所有节点使用同一种聚合函数。

#### Multiplexer

对包含控制边的 MUX 节点，首先聚合控制消息并生成 true/false 两个互斥门：

$$
[g_i^{T},g_i^{F}]
=\operatorname{softmax}\left(
f_{\mathrm{gate}}\left(
\sum_{j\in\mathcal{N}_{\mathrm{ctrl}}(i)}\mathbf{m}_{ji}
\right)
\right).
$$

随后按门控权重组合 true 和 false 数据分支：

$$
\mathbf{a}_{i}^{\mathrm{mux}}
=g_i^{T}\sum_{j\in\mathcal{N}_{T}(i)}\mathbf{m}_{ji}
+g_i^{F}\sum_{j\in\mathcal{N}_{F}(i)}\mathbf{m}_{ji}
+\sum_{j\in\mathcal{N}_{\mathrm{other}}(i)}\mathbf{m}_{ji}.
$$

该机制显式建模 MUX 的互斥选择关系，而不是将两个数据分支无差别相加。

#### Sequential Cells

对寄存器和其他时序节点，编码器使用 reset 和 enable 的嵌套门控：

$$
\mathbf{g}_{i}^{r}=\sigma(f_r(\mathbf{a}_{i}^{\mathrm{reset}})),
\qquad
\mathbf{g}_{i}^{e}=\sigma(f_e(\mathbf{a}_{i}^{\mathrm{enable}})),
$$

$$
\mathbf{a}_{i}^{\mathrm{seq}}
=\mathbf{g}_{i}^{r}\odot\mathbf{a}_{i}^{\mathrm{reset}}
+(1-\mathbf{g}_{i}^{r})\odot
\left[
\mathbf{g}_{i}^{e}\odot\mathbf{a}_{i}^{\mathrm{data}}
+(1-\mathbf{g}_{i}^{e})\odot\mathbf{a}_{i}^{\mathrm{clock}}
\right].
$$

其中 $\odot$ 表示逐元素乘法。该结构表达 reset、enable 和普通数据路径之间的层级关系。

#### Memory Cells

Memory 节点根据目标端口位置学习逐维缩放向量：

$$
\mathbf{a}_{i}^{\mathrm{mem}}
=\sum_{j\in\mathcal{N}_{\mathrm{data}\cup\mathrm{enable}}(i)}
\operatorname{Emb}_{\mathrm{port}}(p_{ji}^{t})
\odot\mathbf{m}_{ji}.
$$

这使不同 memory 端口能够拥有不同的消息贡献。

#### Arithmetic, Comparison, and Shift Cells

对于操作数顺序敏感的节点，分别聚合端口 A 和端口 B：

$$
\mathbf{a}_{i}^{A}=\sum_{j:p_{ji}^{t}=0}\mathbf{m}_{ji},
\qquad
\mathbf{a}_{i}^{B}=\sum_{j:p_{ji}^{t}=1}\mathbf{m}_{ji},
$$

$$
\mathbf{a}_{i}^{\mathrm{pair}}
=f_{\mathrm{pair}}
\left(
[\mathbf{a}_{i}^{A}\Vert\mathbf{a}_{i}^{B}\Vert
\operatorname{Emb}_{\mathrm{op}}(t_i)]
\right),
$$

其中 $t_i$ 区分 arithmetic、comparison 和 shift 三类操作。

#### Other Cells

对于 logic、combinational 及其他节点，模型在注意力加权求和、均值和最大值之间学习类型相关的软选择：

$$
\mathbf{s}_{i}=\sum_{j\in\mathcal{N}(i)}
\alpha_{ji}\mathbf{m}_{ji},
\qquad
\boldsymbol{\mu}_{i}=\frac{1}{|\mathcal{N}(i)|}
\sum_{j\in\mathcal{N}(i)}\mathbf{m}_{ji},
$$

$$
\mathbf{q}_{i}=\max_{j\in\mathcal{N}(i)}\mathbf{m}_{ji},
\qquad
[\beta_i^s,\beta_i^{\mu},\beta_i^q]
=\operatorname{softmax}(\operatorname{Emb}_{\mathrm{agg}}(t_i)),
$$

$$
\mathbf{a}_{i}^{\mathrm{default}}
=\beta_i^s\mathbf{s}_{i}
+\beta_i^{\mu}\boldsymbol{\mu}_{i}
+\beta_i^q\mathbf{q}_{i}.
$$

最终聚合结果 $\mathbf{a}_{i}^{(\ell)}$ 由目标节点的 RTL 类型选择上述分支；当节点存在控制输入时，MUX 聚合具有最高优先级。

### 4. Node and Edge Updates

节点状态通过残差连接、stochastic depth 和 LayerNorm 更新：

$$
\widetilde{\mathbf{h}}_{i}^{(\ell+1)}
=\gamma^{(\ell)}
\left([\mathbf{h}_{i}^{(\ell)}\Vert\mathbf{a}_{i}^{(\ell)}]\right),
$$

$$
\mathbf{h}_{i}^{(\ell+1)}
=\operatorname{LN}\left(
\mathbf{h}_{i}^{(\ell)}
+\operatorname{DropPath}(\widetilde{\mathbf{h}}_{i}^{(\ell+1)})
\right).
$$

边状态只使用更新后的源节点状态和当前边状态：

$$
\widetilde{\mathbf{e}}_{ji}^{(\ell+1)}
=\psi^{(\ell)}
\left([\mathbf{h}_{j}^{(\ell+1)}\Vert\mathbf{e}_{ji}^{(\ell)}]\right),
$$

$$
\mathbf{e}_{ji}^{(\ell+1)}
=\operatorname{LN}\left(
\mathbf{e}_{ji}^{(\ell)}
+\operatorname{DropPath}(\widetilde{\mathbf{e}}_{ji}^{(\ell+1)})
\right).
$$

仅依赖源节点更新边状态，保持了 RTL 信号沿有向连接传播的语义。网络共堆叠 $L$ 个控制感知 GNN 层，且 DropPath 比例随深度线性增加。

### 5. Graph-Level RTL Representation

经过 $L$ 层传播后，节点状态先经过输出投影：

$$
\mathbf{z}_{i}=\mathbf{W}_{o}\mathbf{h}_{i}^{(L)}+\mathbf{b}_{o}.
$$

对属于同一个 RTL 图的全部节点执行全局均值池化，得到图级表示：

$$
\mathbf{z}_{\mathrm{RTL}}
=\frac{1}{|\mathcal{V}|}
\sum_{v_i\in\mathcal{V}}\mathbf{z}_{i}.
$$

$\mathbf{z}_{\mathrm{RTL}}$ 汇总了 cell 类型、信号位宽、边类型、端口角色和控制流语义，是 VeriPCP 进行预仿真覆盖率建模和测试激励排序的 RTL 表征。

### 6. Default Configuration

| Parameter | Default | Description |
| --- | ---: | --- |
| Hidden dimension $D$ | 256 | 节点与边隐状态维度 |
| GNN layers $L$ | 4 | 控制感知消息传递层数 |
| Cell types | 74 | RTL cell-type vocabulary 大小 |
| Edge types | 7 | RTL 依赖边类型数量 |
| Maximum port positions | 8 | 端口位置 embedding 范围 |
| Maximum DropPath rate | 0.1 | 最深 GNN 层的 stochastic-depth 比例 |

### 7. Implementation

RTL encoder 的主要实现位于：

- `models/encoder.py`: `RTLNodeFeatureEncoder`、`RTLEdgeFeatureEncoder` 和 `ControlGatedGNNLayer`；
- `cdfg_rtl/data_types.py`: RTL cell、node 和 edge 类型定义；
- `models/model.py`: RTL encoder 与覆盖率预测模型的集成。
