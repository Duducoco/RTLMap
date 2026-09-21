# RTLMap: Mapping RTL States to Hyperrectangles for Pre-Simulation Stimulus Prioritization in Processor Verification

## RSA-GNN

RTLMap converts processor RTL into a control/data-flow graph (CDFG) and uses a control-aware graph neural network to encode its structural and semantic information. Given an RTL graph

$$
\mathcal{G}_{\mathrm{RTL}}=(\mathcal{V},\mathcal{E}),
$$

each node $v_i\in\mathcal{V}$ represents an RTL cell, and each directed edge $e_{ji}=(v_j,v_i)\in\mathcal{E}$ represents a signal dependency from source node $v_j$ to target node $v_i$. The encoder produces node representations, edge representations, and a graph-level RTL representation for coverage prediction and stimulus prioritization.

### 1. Input Feature Encoding

Each node $v_i$ contains a cell type $c_i$ and signal width $w_i$. Its initial representation is

$$
\mathbf{h}_i^{(0)}=
\operatorname{LN}\left(
\operatorname{Emb}_{\mathrm{cell}}(c_i)
+\mathbf{W}_{w}\log_2(w_i+1)+\mathbf{b}_{w}
\right),
$$

where $\operatorname{Emb}_{\mathrm{cell}}$ is a learnable cell-type embedding and $\operatorname{LN}$ denotes LayerNorm. The logarithmic width transform compresses the wide numerical range of RTL signal widths.

Each edge $e_{ji}$ contains an edge type $r_{ji}$, signal width $w_{ji}$, source-port position $p_{ji}^{s}$, and target-port position $p_{ji}^{t}$. The initial edge representation is

$$
\mathbf{e}_{ji}^{(0)}=
\operatorname{Emb}_{\mathrm{edge}}(r_{ji})
+\mathbf{W}_{e}\log_2(w_{ji}+1)+\mathbf{b}_{e}
+\left[
\operatorname{Emb}_{s}(p_{ji}^{s})\,\Vert\,
\operatorname{Emb}_{t}(p_{ji}^{t})
\right],
$$

where $\Vert$ denotes concatenation. Port positions allow the model to distinguish operand roles, such as input A and input B of a subtractor.

The RTL edge types are `DATA`, `DATA_TRUE`, `DATA_FALSE`, `CONTROL`, `CLOCK`, `RESET`, and `ENABLE`.

### 2. Edge Message Construction

At layer $\ell$, the source-node state and edge state are concatenated. Control edges and other edges use separate message functions:

$$
\mathbf{m}_{ji}^{(\ell)}=
\begin{cases}
\phi_{\mathrm{ctrl}}^{(\ell)}
\left([\mathbf{h}_{j}^{(\ell)}\Vert\mathbf{e}_{ji}^{(\ell)}]\right),
& r_{ji}=\mathrm{CONTROL},\\[4pt]
\phi_{\mathrm{data}}^{(\ell)}
\left([\mathbf{h}_{j}^{(\ell)}\Vert\mathbf{e}_{ji}^{(\ell)}]\right),
& \text{otherwise}.
\end{cases}
$$

Both $\phi_{\mathrm{ctrl}}$ and $\phi_{\mathrm{data}}$ are two-layer MLPs with GELU activations. The encoder also computes an attention weight for each incoming edge:

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

RTL nodes have different circuit semantics. RSA-GNN therefore selects the aggregation path according to the target node type instead of applying one universal aggregation function.

#### Multiplexer Nodes

For a MUX node with control edges, control messages are aggregated to produce two mutually exclusive gates:

$$
[g_i^{T},g_i^{F}]
=\operatorname{softmax}\left(
f_{\mathrm{gate}}\left(
\sum_{j\in\mathcal{N}_{\mathrm{ctrl}}(i)}\mathbf{m}_{ji}
\right)
\right).
$$

The true and false data branches are then combined using the gate weights:

$$
\mathbf{a}_{i}^{\mathrm{mux}}
=g_i^{T}\sum_{j\in\mathcal{N}_{T}(i)}\mathbf{m}_{ji}
+g_i^{F}\sum_{j\in\mathcal{N}_{F}(i)}\mathbf{m}_{ji}
+\sum_{j\in\mathcal{N}_{\mathrm{other}}(i)}\mathbf{m}_{ji}.
$$

This explicitly models the mutually exclusive selection behavior of a MUX rather than adding both data branches indiscriminately.

#### Sequential Cells

For registers and other sequential cells, the encoder uses nested reset and enable gates:

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

Here $\odot$ denotes element-wise multiplication. This represents the priority relationship among reset, enable, and ordinary data paths.

#### Memory Cells

Memory nodes learn a per-dimension scaling vector for each target-port position:

$$
\mathbf{a}_{i}^{\mathrm{mem}}
=\sum_{j\in\mathcal{N}_{\mathrm{data}\cup\mathrm{enable}}(i)}
\operatorname{Emb}_{\mathrm{port}}(p_{ji}^{t})
\odot\mathbf{m}_{ji}.
$$

This allows different memory ports to contribute different messages.

#### Arithmetic, Comparison, and Shift Cells

For cells whose semantics are sensitive to operand order, port A and port B are aggregated separately:

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

where $t_i$ distinguishes arithmetic, comparison, and shift semantics.

#### Other Cells

For logic, combinational, and other cells, the model learns a type-dependent soft mixture of attention-weighted sum, mean, and max aggregation:

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

The final aggregation result $\mathbf{a}_{i}^{(\ell)}$ is selected according to the RTL type of the target node. When a node has control inputs, the MUX aggregation path has the highest priority.

### 4. Node and Edge Updates

Node states are updated with a residual connection, stochastic depth, and LayerNorm:

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

Edge states use the updated source-node state and the current edge state:

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

Updating edges from the source-node state preserves the directionality of RTL signal propagation. The encoder stacks $L$ control-aware GNN layers, with a linearly increasing DropPath rate over depth.

### 5. Graph-Level RTL Representation

After $L$ propagation layers, node states are projected as

$$
\mathbf{z}_{i}=\mathbf{W}_{o}\mathbf{h}_{i}^{(L)}+\mathbf{b}_{o}.
$$

Global mean pooling over all nodes in an RTL graph produces the graph-level representation:

$$
\mathbf{z}_{\mathrm{RTL}}
=\frac{1}{|\mathcal{V}|}
\sum_{v_i\in\mathcal{V}}\mathbf{z}_{i}.
$$

$\mathbf{z}_{\mathrm{RTL}}$ summarizes cell types, signal widths, edge types, port roles, and control-flow semantics. It is the RTL representation used by RTLMap for pre-simulation coverage modeling and stimulus ranking.

### 6. Default Configuration

| Parameter | Default | Description |
| --- | ---: | --- |
| Hidden dimension $D$ | 256 | Dimension of node and edge hidden states |
| GNN layers $L$ | 4 | Number of control-aware message-passing layers |
| Cell types | 74 | Size of the RTL cell-type vocabulary |
| Edge types | 7 | Number of RTL dependency edge types |
| Maximum port positions | 8 | Range of port-position embeddings |
| Maximum DropPath rate | 0.1 | Stochastic-depth rate at the deepest GNN layer |

### 7. Implementation

The main RTL encoder implementation is located in:

- `models/encoder.py`: `RTLNodeFeatureEncoder`, `RTLEdgeFeatureEncoder`, and `ControlGatedGNNLayer`;
- `cdfg_rtl/data_types.py`: RTL cell, node, and edge type definitions;
- `models/model.py`: integration of the RTL encoder with the coverage prediction model.
