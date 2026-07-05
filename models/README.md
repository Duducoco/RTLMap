# 模型说明

当前模型是面向覆盖率报告的双图 GNN。输入为 RTL CDFG 和可选 ASM CDFG，输出为图级覆盖率预测；训练中不再包含 RTL 边覆盖分类任务。

## 当前组件

1. **model.py**
   - `DualGraphFusionModel`：整合 RTL/ASM 编码器、图级覆盖率回归头和可选超矩形头。
   - `create_model`、`create_small_model`、`create_base_model`：模型工厂函数。

2. **encoder.py**
   - `PerceiverDualEncoder`：对 RTL 图和 ASM 图进行编码，并通过 Perceiver 风格跨图融合注入测试激励上下文。
   - RTL 边类型、边宽、端口位置仍作为结构特征参与编码。

3. **heads.py**
   - `GraphRegressor`：按 `coverage_target_keys` 为每类图级覆盖率维护独立回归头。
   - `CoverageHead`：单个覆盖率目标的 MLP 预测头。

4. **hyperrectangle.py**
   - `HyperrectangleHead`：joint contrastive 模式下生成超矩形表示。
   - `hyperrectangle_intersection`：计算两个样本超矩形的交集度。

5. **losses.py**
   - `compute_supervised_losses`：只计算图级覆盖率回归损失。

## 前向输出

```python
output = model(batch)

output.graph_pred      # [B, num_coverage_targets]
output.hyper_min       # [B, hidden_dim]，仅 use_hyperrectangle=True
output.hyper_max       # [B, hidden_dim]，仅 use_hyperrectangle=True
output.rtl_graph_emb   # [B, hidden_dim]，供对比学习使用
```

模型不再输出 `edge_logits`。

## 监督损失

普通训练和 joint contrastive 训练中的监督项都只使用图级覆盖率回归：

```python
losses = model.compute_loss(output, batch)
loss = losses["graph_loss"]
```

损失定义：

```text
L_graph = SmoothL1(graph_pred[valid], graph_target[valid])
L_total = L_graph
```

其中图级覆盖率缺失值为 `NaN`，训练时按目标列独立 mask。

## Joint Contrastive

joint contrastive 模式额外使用覆盖向量相似度监督超矩形交集度：

```text
L_total = lambda_ce * (L_graph(a) + L_graph(b))
        + lambda_cl * L_contrastive(intersection(a, b), similarity(a, b))
```

`similarity(a, b)` 来自 `targets.coverage_vectors` 的完整二值向量 agreement，不使用 merged report，也不使用边分类标签。

## 测试

```bash
uv run python models/test_model.py
uv run --with pytest pytest -q
```

测试覆盖模型前向、图级损失、多覆盖率 head、graph-only 训练步骤和 pair contrastive 训练步骤。
