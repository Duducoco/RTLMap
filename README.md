# RTLMap

RTLMap 是一个面向硬件验证覆盖率预测的双图 GNN 训练项目。项目从 coverage-report-extractor 产出的 manifest 数据集中读取 RTL CDFG 与 ASM CDFG，将它们编码为 PyTorch Geometric 数据，并使用 PyTorch Lightning 训练覆盖率预测模型。

当前仓库的主要能力集中在训练链路，而不是直接生成覆盖率标注数据。输入数据应已按 `manifest.json` 和 `samples.jsonlines` 组织好。

## 功能特性

- **Manifest 数据集加载**：支持 coverage-report-extractor 产出的 `dataset.v1` 普通样本。
- **RTL/ASM 双图建模**：RTL 图提供节点、边、位宽、端口位置等结构特征，ASM 图提供基本块类型与指令编码。
- **双图融合模型**：使用 Perceiver 风格跨图融合，对 RTL 与 ASM 上下文进行交互式编码。
- **图级覆盖率回归**：训练目标为 `branch`、`line`、`fsm`、`toggle`、`condition` 等图级覆盖率。
- **多覆盖率目标**：图级回归可选择 `branch`、`line`、`fsm`、`toggle`、`condition` 子集。
- **CodeBERT 指令编码**：可选启用 HuggingFace CodeBERT 为 ASM 指令生成文本编码；未启用时回退为零向量。
- **缓存与去重**：预处理阶段缓存 RTL/ASM 图结构与 ASM 编码，减少重复构建成本。
- **Joint Contrastive 训练**：使用 `targets.coverage_vectors` 构造 `(a, b)` pair，并用 covered-set positive similarity 监督超矩形对比损失。
- **Lightning 训练封装**：提供 checkpoint、early stopping、TensorBoard/CSV 日志和测试入口。

## 项目结构

```text
RTLMap/
├── main.py                         # 训练/测试 CLI 入口
├── pyproject.toml                  # Python 版本、依赖和 uv 配置
├── run_train.sh                    # 普通训练示例脚本
├── run_train_queue_no_fusion.sh    # pooled-add 无融合模型排队训练
├── run_contrastive.sh              # joint contrastive 训练示例脚本
├── run_contrastive_queue_no_fusion.sh # pooled-add 对比学习排队训练
├── datasets/
│   ├── data_types.py               # DualGraphData、覆盖率目标定义
│   ├── datamodule.py               # 普通 manifest 数据集与 Lightning DataModule
│   ├── pair_datamodule.py          # coverage-vector pair 对比学习数据管线
│   └── coverage_vector_similarity.py # 覆盖向量相似度计算
├── cdfg_rtl/                       # RTL 图枚举和数据类型
├── cdfg_asm/                       # ASM 图枚举和数据类型
├── models/
│   ├── data_types.py               # ModelConfig、ModelOutput
│   ├── encoder.py                  # RTL/ASM 编码器
│   ├── interaction.py              # Perceiver 跨图融合
│   ├── model.py                    # 融合模型、pooled-add baseline 与任务头
│   ├── hyperrectangle.py           # 超矩形表示
│   └── contrastive_loss.py         # 对比损失
├── trainer/
│   ├── config.py                   # TrainerConfig
│   ├── app_config.py               # AppConfig/DataConfig/RuntimeConfig
│   ├── lightning_module.py         # LightningModule 训练逻辑
│   ├── lightning_trainer.py        # Lightning Trainer 包装
│   ├── callbacks.py                # checkpoint/early stopping callbacks
│   └── utils.py                    # train_model 与 DataModule 构建
├── text_encoder/                   # CodeBERT 指令编码
├── metrics/                        # 分类、回归、对比指标
└── tests/                          # pytest 测试
```

## 环境要求

- Python `>=3.12,<3.13`
- uv
- PyTorch `2.2.1+cu121`
- PyTorch Geometric `2.5.0`
- Lightning `>=2.3.0,<=2.4.0`
- Transformers `>=4.40.0,<4.46.0`（仅启用文本编码器时需要下载模型）

安装依赖：

```bash
uv sync
```

查看入口参数：

```bash
uv run python main.py --help
```

## 数据集格式

### 普通训练数据集

普通训练目录需要包含 `manifest.json`，其 `schema_version` 为 `dataset.v1`，并指向样本文件，默认是 `samples.jsonlines`。

```json
{
  "schema_version": "dataset.v1",
  "dataset_name": "example",
  "samples": "samples.jsonlines",
  "rtl_graphs": "rtl_graphs",
  "asm_graphs": "asm_graphs",
  "total": 1,
  "errors": 0
}
```

每行样本使用 `sample.v1`：

```json
{
  "schema_version": "sample.v1",
  "sample_id": "test_a::ALU",
  "test_id": "test_a",
  "module_name": "ALU",
  "rtl_graph": "rtl_graphs/ALU.json",
  "asm_graph": "asm_graphs/test_a.json",
  "targets": {
    "edge_coverage": {
      "default": -1,
      "labels": [
        {"edge_id": 0, "label": 1},
        {"edge_id": 1, "label": 0}
      ]
    },
    "graph_coverage": {
      "unit": "percent",
      "values": {
        "branch": 50.0,
        "line": null,
        "fsm": 0.0,
        "toggle": 100.0,
        "condition": null
      }
    }
  }
}
```

说明：

- `edge_coverage.labels[*].edge_id` 对应 RTL 图中的原始边序号。
- `label=-1` 表示忽略，`0` 表示未覆盖，`1` 表示已覆盖。
- 图级覆盖率以百分比输入，加载时会归一化到 `[0, 1]`。
- 图级覆盖率缺失值使用 `null`，训练时按目标列独立 mask。
- `asm_graph` 可为 `null` 或缺失，加载器会使用一个空 ASM 图占位。

### RTL 图字段

RTL JSON 至少需要包含：

```json
{
  "schema_version": "rtl_graph.v1",
  "module_name": "ALU",
  "nodes": [
    {
      "id": "n0",
      "cell_type": "INPUT",
      "width": 1
    }
  ],
  "edges": [
    {
      "edge_id": 0,
      "source": "n0",
      "target": "n1",
      "source_port_idx": 0,
      "target_port_idx": 0,
      "type": "DATA_TRUE",
      "width": 1
    }
  ]
}
```

加载器会过滤 source/target 不存在的边，并保持过滤后的边标签对齐。

### ASM 图字段

ASM JSON 至少需要包含：

```json
{
  "entry_node": "bb_0",
  "exit_nodes": ["bb_0"],
  "nodes": {
    "bb_0": {
      "id": "bb_0",
      "node_type": "ARITHMETIC",
      "instructions": [
        {"mnemonic": "add", "operands": ["x1", "x2", "x3"]}
      ]
    }
  },
  "edges": []
}
```

## 训练

### 基础训练

```bash
uv run python main.py \
    --dataset-dir /path/to/dataset \
    --data-root ./data \
    --max-epochs 100 \
    --batch-size 16
```

`--dataset-dir` 支持传入多个目录，样本会合并训练：

```bash
uv run python main.py \
    --dataset-dir /path/to/dataset_a /path/to/dataset_b \
    --data-root ./data
```

### 启用 CodeBERT ASM 编码

```bash
uv run python main.py \
    --dataset-dir /path/to/dataset \
    --data-root ./data \
    --use-text-encoder \
    --text-model-name microsoft/codebert-base \
    --text-output-dim 256 \
    --text-max-length 512 \
    --text-pooling mean
```

未传 `--use-text-encoder` 时，ASM 指令编码为零向量，仍可训练结构路径。

### 多覆盖率目标

默认只训练 `branch` 图级回归目标。可显式选择多个目标：

```bash
uv run python main.py \
    --dataset-dir /path/to/dataset \
    --data-root ./data \
    --coverage-targets branch line toggle condition
```

### 仅测试 checkpoint

```bash
uv run python main.py \
    --dataset-dir /path/to/dataset \
    --data-root ./data \
    --test-only \
    --ckpt-path checkpoints/dual_graph_gnn/version_0/checkpoints/best.ckpt
```

## Joint Contrastive 训练

Joint contrastive 模式直接使用新版 `dataset.v1` 普通数据集，不再需要额外构造合并覆盖报告。
数据集中的每条样本需要包含 `targets.coverage_vectors`。

```json
{
  "schema_version": "dataset.v1",
  "dataset_name": "vector_example",
  "samples": "samples.jsonlines",
  "rtl_graphs": "rtl_graphs",
  "asm_graphs": "asm_graphs",
  "total": 2,
  "errors": 0
}
```

训练时，每个有效样本作为 anchor，从同一数据源、同一 `module_name` 的候选中
按 anchor 局部 Coverage Similarity 排名采样 relative-low/mid/high pair。
训练 pair 使用 `seed + epoch` 确定性重采样，同一无序 pair 每个 epoch 最多出现
一次；验证 pair 固定。只要存在不同覆盖签名的同模块候选，分层前就跳过与 anchor
覆盖签名完全相同的候选；仅在全部候选签名都相同时回退。graph 和 volume 这两个 endpoint-local loss 使用 inverse-degree 权重保持
样本等权，IoU calibration 和 ranking loss 按有效 coverage type 等权。

Pair DataLoader 会把本轮已选 pair 按全局 Coverage Similarity 排序，再把完整的
低到高排序蛇形铺到所有 batch，使每个 batch 获得覆盖全分布的等距 rank。DDP
先对全局排序做无重复的交错分片，再在各 rank 内铺排；各 rank 步数一致，尾批
大小可以相差 1。从而在不使用固定 Jaccard 绝对区间的前提下，为 ranking loss
提供足够的目标差。

自动验证集按 `(dataset_dir, test_id)` 整组划分，保证同一个 Test Stimulus 不会
同时出现在训练和验证中，并保证验证 module 在训练集中仍有样本。
每种 coverage type 对应一个五维子矩形。子矩形 log 真实体积监督为该类型的
`covered_count / element_count`，pair 的真实体积 IoU 监督为 covered-set
Jaccard。共同未覆盖的位置不参与相似度：

```text
Covered_type(sample) = {item.id | item.label == 1}
similarity_type(a, b) = |Covered_type(a) ∩ Covered_type(b)|
                        / |Covered_type(a) ∪ Covered_type(b)|
similarity(a, b) = mean(similarity_type(a, b))
```

`element_count == 0` 的 coverage type 不参与体积监督；pair 双方在某类型的
union 为空时，该类型不参与 IoU 监督。空 coverage set 仍通过体积目标学习最小
体积。五个类型的有效 IoU 等权平均，避免稀疏向量中的共同 `0/0` 主导监督。

训练命令：

```bash
uv run python main.py \
    --dataset-dir /path/to/dataset \
    --data-root ./data \
    --joint-contrastive \
    --use-hyperrectangle \
    --contrastive-batch-size 16 \
    --pair-candidate-pool-size 256 \
    --pair-relative-low-quota 2 \
    --pair-relative-mid-quota 1 \
    --pair-relative-high-quota 1 \
    --pair-sampling-seed 42 \
    --lambda-ce 1.0 \
    --lambda-iou 1.0 \
    --iou-rank-loss-weight 0.5 \
    --iou-rank-margin 0.05 \
    --iou-rank-min-target-gap 0.05 \
    --lambda-volume 1.0 \
    --volume-warmup-epochs 5 \
    --smooth-intersection-temperature 0.01 \
    --hyperrectangle-dim-per-type 5
```

可选项：

- `--pair-candidate-pool-size`：每个 anchor 最多检查的同模块候选数。
- `--pair-relative-low-quota`：anchor 局部最低四分位采样数。
- `--pair-relative-mid-quota`：anchor 局部中间区间采样数。
- `--pair-relative-high-quota`：anchor 局部最高四分位采样数。
- `--pair-sampling-seed`：训练 pair 使用 `seed + epoch` 重采样的基础 seed。
- `--lambda-iou`：逐类型真实体积 IoU 对齐权重。
- `--iou-rank-loss-weight`：IoU pairwise ranking 在 IoU 损失中的权重。
- `--iou-rank-margin`：不同目标 pair 的最小预测 IoU 排序间隔。
- `--iou-rank-min-target-gap`：参与排序监督的最小目标 Jaccard 差。
- `--lambda-volume`：单样本真实体积校准权重。
- `--volume-warmup-epochs`：体积权重线性 warmup epoch 数。
- `--smooth-intersection-temperature`：仅训练阶段使用的平滑交集温度。
## 常用 CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset-dir` | 必需 | 一个或多个 manifest 数据集目录 |
| `--data-root` | 必需 | 预处理缓存根目录 |
| `--model-architecture` | `perceiver_fusion` | `perceiver_fusion` 或无融合的 `pooled_add` |
| `--hidden-dim` | `256` | 模型隐藏维度 |
| `--num-gnn-layers` | `6` | GNN 层数 |
| `--dropout` | `0.1` | dropout |
| `--max-epochs` | `100` | 最大训练 epoch |
| `--lr` | `1e-4` | 学习率 |
| `--weight-decay` | `1e-5` | AdamW weight decay |
| `--batch-size` | `16` | 普通 DataLoader batch 大小 |
| `--num-workers` | `4` | 数据加载 worker 数 |
| `--precision` | `bf16-mixed` | Lightning precision |
| `--logger-type` | `tensorboard` | `tensorboard` 或 `csv` |
| `--checkpoint-dir` | `checkpoints` | checkpoint 和日志根目录 |
| `--use-text-encoder` | `false` | 启用 CodeBERT ASM 编码 |
| `--coverage-targets` | `branch` | 图级覆盖率目标 |
| `--use-hyperrectangle` | `false` | 启用超矩形表示 |
| `--joint-contrastive` | `false` | 启用 coverage-vector pair 对比训练 |
| `--hyperrectangle-dim-per-type` | `10` | 每种 coverage type 的子矩形维度 |
| `--fast-dev-run` | `false` | Lightning 快速调试模式 |
| `--seed` | `None` | 全局随机种子 |

### 无融合 pooled-add baseline

`pooled_add` 保留 RTL 和 ASM 各自的 GNN 编码器，但不创建 Perceiver
cross-attention。两路节点表示分别执行 mean pooling，然后逐元素相加：

```text
joint_graph = mean_pool(rtl_node) + mean_pool(asm_node)
```

回归 head、可选 hyperrectangle head 及其维度和 margin 配置与融合模型一致。
单次训练可通过 `--model-architecture pooled_add` 选择。依次训练四个数据集：

```bash
bash run_train_queue_no_fusion.sh
```

该队列脚本复用 `run_train.sh` 的仅监督配置，默认输出到
`checkpoints/no_fusion`，实验名为 `<dataset>-4coverage-no-fusion`。

使用相同无融合架构依次运行四个 joint contrastive 实验：

```bash
bash run_contrastive_queue_no_fusion.sh
```

该脚本复用 `run_contrastive.sh` 的图回归、Hyperrectangle、volume loss 和 IoU
loss 配置，默认输出到 `checkpoints/contrastive_no_fusion`。任一数据集训练失败时
队列会立即停止，不会继续执行后续实验。

注意：`TrainerConfig` 中还存在 `use_bucketing`、`token_budget`、`strategy` 等配置，目前未在 `main.py` CLI 暴露。如需使用，可通过 Python API 构造配置。

## Python API

```python
from models import ModelConfig
from trainer import AppConfig, DataConfig, RuntimeConfig, TrainerConfig, train_model

app_config = AppConfig(
    data=DataConfig(
        data_root="./data",
        dataset_dir="/path/to/dataset",
    ),
    model=ModelConfig(
        model_architecture="pooled_add",
        hidden_dim=256,
        num_gnn_layers=6,
        coverage_target_keys=("branch",),
    ),
    trainer=TrainerConfig(
        max_epochs=100,
        batch_size=16,
        learning_rate=1e-4,
    ),
    text_encoder=None,
    runtime=RuntimeConfig(
        experiment_name="dual_graph_gnn",
        logger_type="tensorboard",
    ),
)

module, lightning_trainer = train_model(app_config)
```

## 测试

运行完整测试：

```bash
uv run python -m pytest -q
```

也可以运行指定测试：

```bash
uv run python -m pytest -q tests/test_manifest_dataset.py
uv run python -m pytest -q tests/test_multi_coverage_heads.py
uv run python -m pytest -q tests/test_pair_contrastive_training.py
```

当前测试重点覆盖：

- manifest 数据集加载与稀疏标签处理
- 多 dataset 目录合并
- coverage-vector pair 相似度与 joint train DataModule 构建
- 多覆盖率 graph heads
- pair 对比训练步骤

## 缓存与输出

- `data_root/train/processed/`：普通训练样本缓存。
- `data_root/train/processed/rtl_*.pt`：去重后的 RTL 图结构缓存。
- `data_root/train/processed/asm_*.pt`：去重后的 ASM 图缓存。
- `data_root/asm_encoding_cache/`：CodeBERT pooled ASM 编码缓存。
- `checkpoint-dir/experiment-name/`：checkpoint 文件，以及 logger 的 `version_xx/` 子目录。
- `checkpoint-dir/experiment-name/version_xx/eval_metric.json`：每个验证 epoch 的评估指标 JSON 导出。

如果数据集内容或编码器配置变化，建议使用新的 `--data-root`，避免复用旧缓存导致实验不一致。

## 已知边界

- 本仓库当前 README 描述的是训练侧；原始 RTL/覆盖率报告提取工具不在当前代码树中作为主入口维护。
- 普通训练默认会在没有显式 val cache 时，从 train 数据集中按固定随机种子划分 9:1 验证集。
- `main.py` 暂未暴露所有 `TrainerConfig` 字段，高级训练配置建议使用 Python API。
- 启用 `--use-text-encoder` 需要可访问 HuggingFace 模型权重，离线环境需提前准备缓存。
