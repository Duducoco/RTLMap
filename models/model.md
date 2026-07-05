# 图级覆盖率回归模型

本文档描述当前 `models/` 目录中的模型实现。当前版本已经移除边覆盖分类任务，只保留图级覆盖率回归和可选的超矩形对比学习。

## 1. 模型目标

给定一个 RTL CDFG 和对应测试的 ASM CDFG，模型预测模块级覆盖率：

```text
branch, line, fsm, toggle, condition
```

训练信号来自：

- `targets.graph_coverage`：图级覆盖率回归目标。
- `targets.coverage_vectors`：joint contrastive 模式下计算覆盖报告相似度。

## 2. 数据流

```text
RTL CDFG ──┐
           ├── PerceiverDualEncoder ── rtl_graph_emb ── GraphRegressor ── graph_pred
ASM CDFG ──┘                                  │
                                              └── HyperrectangleHead ── hyper_min / hyper_max
```

RTL CDFG 提供节点、边、位宽、端口位置等结构信息。ASM CDFG 提供测试激励上下文。编码阶段使用双图融合；预测阶段只使用 RTL 图级表示。

## 3. RTL 消息传递

RTL 编码器仍然使用边结构特征，包括：

- edge type
- edge width
- source port index
- target port index

这些特征用于构造更准确的 RTL 表示，但不再作为边分类预测目标。

## 4. ASM ↔ RTL 融合

`PerceiverDualEncoder` 使用 Perceiver 风格跨图融合，让 RTL 表示读取 ASM 测试激励上下文。

核心目的：

- 从 ASM 中提炼影响硬件覆盖的测试行为。
- 将测试上下文注入 RTL 节点和图级表示。
- 避免直接构造 merged coverage report。

## 5. 图级回归

`GraphRegressor` 对每个覆盖率目标维护独立 head：

```text
graph_pred[k] = CoverageHead_k(rtl_graph_emb)
```

如果配置为：

```python
coverage_target_keys = ("branch", "line", "toggle")
```

则输出形状为：

```text
graph_pred: [B, 3]
```

## 6. 监督损失

图级标签中的缺失值使用 `NaN` 表示。损失按覆盖率列独立 mask：

```text
valid = ~isnan(target)
L_graph = SmoothL1(graph_pred[valid], target[valid])
L_total = L_graph
```

当前普通训练和 joint contrastive 训练都不包含边分类 loss。

## 7. 超矩形对比学习

启用 `use_hyperrectangle` 后，模型从 `rtl_graph_emb` 生成：

```text
hyper_min: [B, D]
hyper_max: [B, D]
```

对两个样本 `(a, b)`，计算超矩形交集度：

```text
intersection(a, b)
```

然后与覆盖向量相似度对齐。

## 8. 覆盖向量相似度

joint contrastive 模式使用 `targets.coverage_vectors`，按固定顺序拼接：

```text
line, condition, toggle, fsm, branch
```

相似度定义：

```text
Covered(sample) = {offset[type] + item.id | item.label == 1}
similarity(a, b) = 1 - |Covered(a) △ Covered(b)| / N
```

这里 `N` 是拼接后总长度。未出现在 `items` 中的位置视为明确的 0，因此大量 0 不会被丢弃，0/0 匹配也会提升相似度。

## 9. Joint Loss

对 pair batch：

```text
out_a = model(batch_a)
out_b = model(batch_b)
```

监督项：

```text
L_sup = L_graph(a) + L_graph(b)
```

对比项：

```text
L_cl = contrastive_loss(intersection(out_a, out_b), similarity(a, b))
```

总损失：

```text
L_total = lambda_ce * L_sup + lambda_cl * L_cl
```

## 10. 当前输出接口

`ModelOutput` 当前主要字段：

```python
graph_pred
rtl_final
asm_final
hyper_min
hyper_max
rtl_graph_emb
```

不再包含 `edge_logits`。
