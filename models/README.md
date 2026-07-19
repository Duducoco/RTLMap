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
   - `HyperrectangleHead`：从 RTL+ASM 图级摘要为五种 coverage type 分别生成可配置维度的 center-radius 子矩形。
   - `hyperrectangle_geometry`：计算逐类型硬或平滑交集、真实体积和真实体积 IoU。

5. **losses.py**
   - `compute_supervised_losses`：只计算图级覆盖率回归损失。

## 前向输出

```python
output = model(batch)

output.graph_pred      # [B, num_coverage_targets]
output.hyper_min       # [B, 5, D_box]，仅 use_hyperrectangle=True
output.hyper_max       # [B, 5, D_box]，仅 use_hyperrectangle=True
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

joint contrastive 模式额外使用 coverage density 和 Jaccard 监督真实体积几何：

```text
L_total = lambda_ce * (L_graph(a) + L_graph(b)) / 2
        + lambda_iou * (L_iou_calibration + w_rank * L_iou_rank)
        + warmed_lambda_volume * SmoothL1(log(box_volume), log(coverage_density))
```

五种 coverage type 分开计算，再对有效类型等权平均。排序项忽略目标差小于阈值
或完全相同的 pair。训练使用平滑交集和 log-IoU，验证与推理使用硬交集；不使用
merged report，也不使用边分类标签。

Pair batch 从当前 Geometry Pair Set 的完整 Coverage Similarity 排序中取得等距
rank。日志中的 `iou_rank_informative_pairs_per_batch` 和
`iou_rank_active_types_per_batch` 用于确认每批平均有多少个有效比较和覆盖类型，
batch 内实际存在可用的排序监督。

## 测试

```bash
uv run python models/test_model.py
uv run --with pytest pytest -q
```

测试覆盖模型前向、图级损失、多覆盖率 head、graph-only 训练步骤和 pair contrastive 训练步骤。
