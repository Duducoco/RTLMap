# 数据集描述（学术论文用）

本文档整理从 `analyze_dataset.py` 全量分析中提取的关键统计数据，按学术论文的 **Dataset / Experimental Setup** 章节结构组织，提供可直接引用的表格、数值和描述文本。

---

## 1. 论文中应包含的核心内容

### 1.1 建议的表格

| 表编号 | 内容 | 用途 |
|--------|------|------|
| Table I | Dataset Overview | 数据集规模、benchmark 信息 |
| Table II | RTL Graph Statistics per Module | 每模块图规模与覆盖率分布 |
| Table III | Edge Label Distribution | 类别不平衡分析（核心挑战） |
| Table IV | ASM Graph Statistics | 测试激励图的规模特征 |
| Table V | Per-Module Structural Profile | 逐模块结构指纹（MUX%、cell\_type 组成） |
| Table VI | Per-Module Edge-Type Label Distribution | 按边类型的标签分布差异 |
| Table VII | Coverage Variation by Module | 有变化模块的覆盖率区间分布 |

### 1.2 建议的图表

| 图编号 | 内容 | 用途 |
|--------|------|------|
| Fig. X | Coverage distribution histogram (by module) | 展示覆盖率分布的多模态特性 |
| Fig. X | Edge label pie chart (全部 vs 有效) | 直观展示标签稀疏性和类别不平衡 |
| Fig. X | RTL node type distribution | 展示 MUX 主导的图结构特征 |

---

## 2. Table I: Dataset Overview

> **写作建议**：放在 Section IV (Experimental Setup) 开头，一段话 + 一个表。

### 表格草稿

| Item | Value |
|------|-------|
| Benchmark Design | CV32E40P (RISC-V RV32IMCXpulp) |
| RTL Modules | 9 |
| Test Cases | 6,979 |
| RTL Graph Samples | 62,804 (6,979 × 9) |
| ASM CDFG Samples | 4,995 |
| Total RTL Edges | 70,384,074 |
| Total RTL Nodes | 22,052,580 |
| Synthesis Tool | Yosys + slang plugin |
| Coverage Source | VCS/URG branch coverage reports |

### 描述文本草稿

> We evaluate our approach on the CV32E40P, an open-source 4-stage in-order RISC-V processor core implementing the RV32IMCXpulp ISA, which is widely used as a verification benchmark in the EDA community. The design is decomposed into 9 RTL modules after synthesis (Table I). We collected 6,979 constrained-random test cases, each producing a RISC-V assembly program and per-module VCS/URG branch coverage reports. The RTL control-data flow graphs (CDFGs) are extracted from Yosys RTLIL JSON, yielding a total of 62,804 RTL graph instances. Assembly-level CDFGs are constructed at the basic-block granularity from the test stimulus programs, resulting in 4,995 paired ASM graphs.

---

## 3. Table II: RTL Graph Statistics per Module

> **写作建议**：紧跟 Table I 之后，突出图规模差异和覆盖率分布特点。

### 表格草稿

| Module | \|V\| | \|E\| | Density | Branch Cov. (%) | σ |
|--------|-------|-------|---------|-----------------|---|
| aligner | 93 | 262 | 0.030 | 83.33 | 0.00 |
| alu\_div | 77 | 187 | 0.032 | 56.52 | 0.00 |
| compressed\_decoder | 143 | 476 | 0.023 | 75.00 | 0.00 |
| controller | 993 | 2,938 | 0.003 | 65.65 | 16.30 |
| decoder | 1,480 | 5,193 | 0.002 | 13.49 | 1.88 |
| ff\_one | 68 | 188 | 0.041 | 100.00 | 0.00 |
| int\_controller | 54 | 149 | 0.051 | 8.57 | 0.00 |
| mult | 120 | 287 | 0.020 | 77.78 | 0.00 |
| register\_file | 133 | 409 | 0.023 | 100.00 | 0.00 |

### 关键发现（可写入论文）

1. **图拓扑固定，标签变化**：由于所有测试用例共享相同的 RTL 设计，每个模块的图结构（|V|, |E|）在所有样本间完全一致。样本间的差异完全体现在边的覆盖率标签上，这使得模型必须学会从 ASM 图的差异中推断 RTL 边的覆盖状态。

2. **模块规模跨度大**：最小模块 int\_controller 仅 54 节点，最大模块 decoder 达 1,480 节点（27× 差异），图密度从 0.002 到 0.051 不等，对 GNN 的多尺度适应性提出要求。

3. **覆盖率分布异质性**：7/9 模块的覆盖率在测试用例间无变化（σ=0），仅 controller（σ=16.3, 范围 35.2–81.5%）和 decoder（σ=1.88, 范围 10.9–17.7%）表现出测试敏感性。这意味着图级回归任务的有效训练信号集中在少数模块上。

### 描述文本草稿

> As shown in Table II, the RTL graph topology is deterministic per module — all 6,979 samples of a given module share identical node and edge counts, with variation arising solely from the per-edge coverage labels. Module sizes span a 27× range, from 54 nodes (int\_controller) to 1,480 nodes (decoder), with graph densities varying from 0.002 to 0.051. Notably, only 2 out of 9 modules exhibit non-trivial coverage variation across test cases: the controller (σ=16.3%) and decoder (σ=1.88%), while the remaining 7 modules have constant branch coverage regardless of the test stimulus applied.

---

## 4. Table III: Edge Label Distribution（核心表格）

> **写作建议**：这是论文中最重要的数据表之一，直接揭示了模型训练面临的核心挑战。建议配合一段"Class Imbalance Analysis"讨论。

### 全局分布

| Label | Count | Proportion |
|-------|-------|------------|
| −1 (unlabeled) | 42,531,079 | 60.4% |
| 0 (not covered) | 898,287 | 1.3% |
| 1 (covered) | 26,954,708 | 38.3% |
| **Total** | **70,384,074** | **100%** |

### 有效标签分布（仅 0 和 1）

| Label | Count | Proportion |
|-------|-------|------------|
| 0 (not covered) | 898,287 | 3.2% |
| 1 (covered) | 26,954,708 | 96.8% |

**正负样本比 = 30:1**（已覆盖:未覆盖）

### 按模块分解

| Module | Labeled Edges | Label=1 | Label=0 | Ratio (1:0) | Unlabeled Rate |
|--------|---------------|---------|---------|-------------|----------------|
| aligner | 1,395,800 | 1,381,842 | 13,958 | 99.0:1 | 23.7% |
| decoder | 24,677,550 | 24,665,596 | 11,954 | 2,063:1 | 31.9% |
| int\_controller | 935,186 | 279,160 | 656,026 | 0.43:1 | 10.1% |
| mult | 844,459 | 628,110 | 216,349 | 2.9:1 | 57.8% |
| alu\_div | 0 | 0 | 0 | — | 100% |
| compressed\_decoder | 0 | 0 | 0 | — | 100% |
| controller | 0 | 0 | 0 | — | 100% |
| ff\_one | 0 | 0 | 0 | — | 100% |
| register\_file | 0 | 0 | 0 | — | 100% |

### 关键发现

1. **标签稀疏性 (Label Sparsity)**：60.4% 的边缺乏覆盖率标注，有效标签率仅 39.6%。在 62,804 个 RTL 样本中，34,892 个（55.6%）的所有边均为 −1（无有效标注），完全无法提供边级监督信号。

2. **极端类别不平衡 (Extreme Class Imbalance)**：在有效标签中，正样本（covered）与负样本（not covered）的比值高达 30:1。更极端地，decoder 模块达到 2,063:1。这意味着朴素的交叉熵损失将严重偏向多数类。

3. **模块级标注不均匀**：仅 4/9 模块（aligner, decoder, int\_controller, mult）具有边级覆盖标注；其余 5 个模块虽有图级 `branch_coverage` 值，但无边级标签，无法提供边分类的训练信号。

4. **唯一的负样本富集模块**：int\_controller 是唯一一个负样本多于正样本的模块（0.43:1），是检验模型对未覆盖边检测能力的关键测试模块。

### 描述文本草稿

> **Class imbalance.** The dataset exhibits a pronounced label sparsity and class imbalance that poses significant challenges for edge-level coverage prediction. As detailed in Table III, 60.4% of all edges are unlabeled (coverage\_label = −1), meaning that no ground-truth coverage information could be mapped to these edges during annotation. Among the 27.85M edges with valid labels, the distribution is heavily skewed toward the covered class: 96.8% are labeled as covered (label=1) versus only 3.2% as not covered (label=0), yielding a positive-to-negative ratio of approximately 30:1.
>
> Furthermore, only 4 out of 9 modules (aligner, decoder, int\_controller, mult) carry edge-level annotations, while the remaining 5 modules have graph-level branch coverage values but no per-edge labels. Among the annotated modules, the imbalance ratio varies dramatically — from 0.43:1 in int\_controller (more uncovered than covered edges) to 2,063:1 in decoder. This heterogeneous label distribution necessitates module-aware loss weighting and robust evaluation metrics beyond accuracy.
>
> To address these challenges, we employ: (i) focal loss with class-specific weighting to handle the 30:1 imbalance; (ii) masking of unlabeled edges (label=−1) during loss computation; and (iii) graph-level regression as an auxiliary task to provide supervision for modules lacking edge-level labels.

---

## 5. Table IV: ASM Graph Statistics

> **写作建议**：简短描述测试激励图的特征，强调与 RTL 图的互补性。

### 表格草稿

| Statistic | Min | Max | Mean | Median | Std |
|-----------|-----|-----|------|--------|-----|
| Basic blocks | 142 | 16,867 | 6,722 | 3,464 | 5,952 |
| Edges | 141 | 55,534 | 20,863 | 11,209 | 20,613 |
| Instructions | 4,737 | 88,119 | 35,560 | 16,403 | 30,796 |
| Instr. per block | 1 | 11,339 | 5.29 | 2 | — |

### ASM 节点类型分布

| Node Type | Count | Proportion |
|-----------|-------|------------|
| BRANCH | 13,196,776 | 39.3% |
| JUMP | 9,526,997 | 28.4% |
| ARITHMETIC | 5,135,781 | 15.3% |
| LOGIC | 1,714,763 | 5.1% |
| SHIFT | 1,461,791 | 4.4% |
| Others | 2,546,569 | 7.5% |

### 描述文本草稿

> The assembly-level CDFGs exhibit high variability in size, with basic block counts ranging from 142 to 16,867 (median=3,464) and total instruction counts from 4,737 to 88,119 (median=16,403). This 119× range in graph size reflects the diversity of the constrained-random test generation strategy. The ASM node type distribution is dominated by control-flow nodes: BRANCH (39.3%) and JUMP (28.4%) together account for 67.7% of all basic blocks, followed by ARITHMETIC (15.3%) and LOGIC (5.1%). Data dependency edges constitute 74.7% of all ASM edges, with branch-related control edges making up the remaining 23.6%.

---

## 6. RTL 图节点类型分布

> **写作建议**：可放在 "Graph Representation" 子节或附录中，说明 CDFG 的组成特征。

### 表格草稿

| Node Type | Count | Proportion | Description |
|-----------|-------|------------|-------------|
| MUX | 14,301,552 | 64.9% | Control-flow multiplexers |
| COMPARE | 3,180,933 | 14.4% | Comparison operators |
| LOGIC | 2,023,366 | 9.2% | Boolean logic |
| OUTPUT | 955,748 | 4.3% | Module output ports |
| INPUT | 914,077 | 4.1% | Module input ports |
| SEQUENTIAL | 307,040 | 1.4% | Flip-flops / registers |
| ARITHMETIC | 174,473 | 0.8% | Arithmetic operators |
| CONSTANT | 174,454 | 0.8% | Constant values |
| SHIFT | 20,937 | 0.1% | Shift operators |

### RTL 边类型分布

| Edge Type | Count | Proportion |
|-----------|-------|------------|
| DATA\_TRUE | 18,745,566 | 26.6% |
| DATA | 18,186,722 | 25.8% |
| CONTROL | 17,580,272 | 25.0% |
| DATA\_FALSE | 15,257,434 | 21.7% |
| ENABLE | 307,040 | 0.4% |
| CLOCK | 307,040 | 0.4% |

### 描述文本草稿

> The RTL CDFGs are dominated by multiplexer (MUX) nodes, which constitute 64.9% of all nodes across the 9 modules (Table X). MUX nodes are the primary carriers of branch coverage information, as each MUX corresponds to a conditional construct (if-else, case, or ternary operator) in the RTL source. The MUX-centric structure produces a near-uniform distribution of edge types: DATA\_TRUE (26.6%), generic DATA (25.8%), CONTROL (25.0%), and DATA\_FALSE (21.7%), reflecting the three-port MUX topology (select, true-branch, false-branch).

---

## 7. 关键统计数据速查（可直接引用的数值）

供行文中引用的关键数字，无需单独成表：

- **数据集规模**：6,979 test cases × 9 modules = 62,804 RTL samples；4,995 ASM samples
- **图规模**：RTL 54–1,480 nodes, 149–5,193 edges；ASM 142–16,867 blocks, 4,737–88,119 instructions
- **标签有效率**：39.6%（有效标注边占总边数）
- **类别不平衡比**：30:1 (covered vs. not covered)
- **有边级标注的模块数**：4/9
- **全部为 −1 的样本占比**：55.6%（34,892/62,804）
- **覆盖率变化最大的模块**：controller（mean=65.65%, σ=16.3%）
- **覆盖率最低的模块**：int\_controller（mean=8.57%, σ=0）
- **ASM-RTL 覆盖率相关性**：Pearson r = 0.053（弱相关）
- **RTL 图密度**：mean=0.03, range 0.002–0.051
- **MUX 节点占比**：64.9%（RTL 图中的主导节点类型）

---

## 8. 逐模块详细画像（Per-Module Profile）

> **写作建议**：可作为论文附录的 Table 或用于 supplementary material。以下提炼了每个模块的结构指纹和覆盖率特征。

### Table V: Per-Module Structural Profile

| Module | \|V\| | \|E\| | Density | Avg. Deg. | MUX% | Top Cell Types |
|--------|-------|-------|---------|-----------|------|----------------|
| aligner | 93 | 262 | 0.030 | 5.63 | 59.1% | $mux(50), input(9), $eq(6), $aldff(6) |
| alu\_div | 77 | 187 | 0.032 | 4.86 | 40.3% | $mux(26), $logic\_not(15), input(10), $aldff(8) |
| compressed\_decoder | 143 | 476 | 0.023 | 6.66 | 73.4% | $mux(94), $eq(11), $pmux(11), output(10) |
| controller | 993 | 2,938 | 0.003 | 5.92 | 73.9% | $mux(693), input(64), output(44), $pmux(41) |
| decoder | 1,480 | 5,193 | 0.002 | 7.02 | 68.0% | $mux(945), $eq(395), output(137), $pmux(62) |
| ff\_one | 68 | 188 | 0.041 | 5.53 | 55.9% | $mux(33), $reduce\_or(14), $eq(5), input(5) |
| int\_controller | 54 | 149 | 0.051 | 5.52 | 44.4% | $mux(16), $or(8), $pmux(8), $logic\_or(3) |
| mult | 120 | 287 | 0.020 | 4.78 | 38.3% | $mux(39), $mul(8), $add(7), $eq(7) |
| register\_file | 133 | 409 | 0.023 | 6.15 | 72.9% | $mux(90), $eq(7), output(5), $pmux(4) |

### 关键发现

1. **MUX 占比差异揭示模块功能特征**：
   - 高 MUX 模块（controller 73.9%, compressed\_decoder 73.4%, register\_file 72.9%）：控制逻辑密集，分支决策多，是覆盖率标注的主要目标
   - 低 MUX 模块（mult 38.3%, alu\_div 40.3%）：数据路径为主，运算密集（$mul, $add 等），分支覆盖的信息密度较低

2. **平均度稳定在 5–7**：所有模块的平均节点度（2E/N）集中在 4.78–7.02 之间，表明 GNN 的消息传递范围在 2–3 跳即可覆盖大部分局部结构

3. **图密度的两个极端**：
   - decoder（0.002）和 controller（0.003）：大规模稀疏图，需要较深的 GNN 层数或全局注意力机制
   - int\_controller（0.051）和 ff\_one（0.041）：小而密，局部信息充足

### Table VI: Per-Module Edge-Type Label Distribution (单样本)

> 展示同一模块内不同边类型的标签分布差异，揭示覆盖率标注的边类型选择性。

**aligner**（有标签）：

| Edge Type | Count | Labeled Rate | Label=1 | Label=0 |
|-----------|-------|-------------|---------|---------|
| CONTROL | 69 | 100% | 68 | 1 |
| DATA\_TRUE | 73 | 100% | 72 | 1 |
| DATA\_FALSE | 58 | 100% | 58 | 0 |
| DATA | 50 | 0% | — | — |
| CLOCK | 6 | 0% | — | — |
| ENABLE | 6 | 0% | — | — |

**int\_controller**（唯一负样本主导模块）：

| Edge Type | Count | Labeled Rate | Label=1 | Label=0 |
|-----------|-------|-------------|---------|---------|
| CONTROL | 28 | 100% | 4 | 24 |
| DATA\_TRUE | 35 | 100% | 24 | 11 |
| DATA\_FALSE | 30 | 100% | 12 | 18 |
| DATA | 48 | 0% | — | — |
| CLOCK | 4 | 0% | — | — |
| ENABLE | 4 | 0% | — | — |

**decoder**（极端正偏模块）：

| Edge Type | Count | Labeled Rate | Label=1 | Label=0 |
|-----------|-------|-------------|---------|---------|
| CONTROL | 1,180 | 100% | ~1,178 | ~2 |
| DATA\_TRUE | 1,432 | 100% | ~1,431 | ~1 |
| DATA\_FALSE | 1,064 | 100% | ~1,063 | ~1 |
| DATA | 1,359 | 0% | — | — |
| CLOCK | 79 | 0% | — | — |
| ENABLE | 79 | 0% | — | — |

### 关键发现

1. **标签仅附着于 MUX 相关边**：CONTROL、DATA\_TRUE、DATA\_FALSE 三种边类型的标注率为 100%，而 DATA、CLOCK、ENABLE 始终未标注。这符合分支覆盖的语义——覆盖信息自然映射到 MUX 的控制与数据端口。

2. **int\_controller 的负样本来源**：CONTROL 边中 24/28 为未覆盖（85.7%），DATA\_FALSE 中 18/30 为未覆盖（60%）。这说明该模块的中断控制逻辑大量分支路径未被测试激励触发。

3. **decoder 的标签几乎全为正**：在 3,676 条有标签边中仅约 4 条为未覆盖。这意味着在该模块上训练边分类器面临约 900:1 的局部不平衡。

### Table VII: Coverage Variation by Module

> 仅 controller 和 decoder 有 σ>0 的覆盖率变化，以下为其区间分布（全量 6,977/6,975 个样本）。

**controller**（σ=16.3%）：

| Coverage Range | Count | Proportion |
|----------------|-------|------------|
| [20%, 40%) | ~1,300 | ~18.6% |
| [40%, 60%) | ~1,700 | ~24.4% |
| [60%, 80%) | ~1,100 | ~15.8% |
| [80%, 100%] | ~2,900 | ~41.2% |

**decoder**（σ=1.88%）：

| Coverage Range | Count | Proportion |
|----------------|-------|------------|
| [10%, 12%) | ~350 | ~5.0% |
| [12%, 14%) | ~4,900 | ~70.3% |
| [14%, 16%) | ~1,350 | ~19.4% |
| [16%, 18%) | ~375 | ~5.4% |

### 描述文本草稿

> **Per-module structural analysis** reveals substantial heterogeneity in both graph topology and coverage characteristics (Table V). MUX nodes, which carry branch coverage information, range from 38.3% of all nodes in the mult module to 73.9% in the controller, reflecting the diversity of computational versus control-logic emphasis across modules. Coverage labels are exclusively associated with MUX-related edge types (CONTROL, DATA\_TRUE, DATA\_FALSE), as shown in Table VI, while generic DATA, CLOCK, and ENABLE edges remain unlabeled across all modules.
>
> Among the 4 modules with edge-level annotations, the int\_controller stands out as the only module where uncovered edges (label=0) outnumber covered ones (label=1), with a ratio of 0.43:1. This module's interrupt handling logic contains many conditional branches not triggered by the test suite, making it a critical benchmark for evaluating the model's ability to detect uncovered paths. In contrast, the decoder module exhibits a label ratio of 2,063:1 (covered:uncovered), presenting an extreme local imbalance challenge.
>
> Coverage variation across test cases is concentrated in two modules: the controller (σ=16.3%, range 35.2–81.5%) shows a broad, roughly bimodal distribution skewed toward high coverage, while the decoder (σ=1.88%, range 10.9–17.7%) varies within a narrow band centered at 13.3%. The remaining 7 modules produce constant branch coverage regardless of the test stimulus, yielding zero training signal for graph-level regression in those modules.

---

## 9. 论文写作建议

### 9.1 应重点讨论的挑战

1. **标签稀疏性与类别不平衡**：这是本数据集的核心挑战，应在 Introduction 和 Method 中均有体现。建议使用 focal loss、class weighting、或 cost-sensitive learning。

2. **图拓扑不变性**：同一模块的所有样本共享完全相同的图结构，差异仅在标签。这一特性可以从两方面论述：
   - 挑战：模型无法从图结构差异中学习，必须依赖 ASM 图的跨图信息传递
   - 优势：消除了图结构变异的混淆因素，使得模型的预测能力完全来自对测试激励（ASM）的理解

3. **模块间异质性**：5/9 模块无边级标签，覆盖率分布差异巨大（8.57%–100%）。建议讨论 transfer learning 或 module-agnostic representation 的策略。

4. **ASM-RTL 弱相关性**：Pearson r=0.053 表明 ASM 图规模与 RTL 覆盖率几乎无线性相关，说明模型需要捕捉更深层的语义关系而非简单的规模映射。

### 9.2 评价指标建议

鉴于极端不平衡，应避免使用 accuracy 作为主要指标。推荐：

- **边分类**：F1-score (尤其是 F1 of class 0)、AUROC、AUPRC、Matthews Correlation Coefficient (MCC)
- **图回归**：MAE、RMSE、R²（按模块分组报告）
- **按模块分组报告**：int\_controller（负样本主导）和 decoder（正样本主导）的分别表现

### 9.3 描述语言规范

- benchmark 名称：CV32E40P（首次出现用全称 "CV32E40P, a 4-stage in-order RISC-V processor core"）
- 图类型：RTL 图用 "CDFG (Control-Data Flow Graph)"，ASM 图用 "assembly-level CDFG" 或 "basic-block-level control flow graph"
- 覆盖率类型：始终明确为 "branch coverage"（非 line/toggle/FSM coverage）
- 标签含义：−1 = "unlabeled"（not "unknown"），0 = "not covered"，1 = "covered"
