# RTLMap

RTLMap 是一个面向硬件验证覆盖率预测的双图 GNN 训练项目。项目从 coverage-report-extractor 产出的 manifest 数据集中读取 RTL CDFG 与 ASM CDFG，将它们编码为 PyTorch Geometric 数据，并使用 PyTorch Lightning 训练覆盖率预测模型。

当前仓库的主要能力集中在训练链路，而不是直接生成覆盖率标注数据。输入数据应已按 `manifest.json`、`samples.jsonlines` 或 `contrastive_samples.jsonlines` 组织好。

## 功能特性

- **Manifest 数据集加载**：支持 `dataset.v1` 普通样本与 `contrastive_dataset.v1` 三元组样本。
- **RTL/ASM 双图建模**：RTL 图提供节点、边、位宽、端口位置等结构特征，ASM 图提供基本块类型与指令编码。
- **双图融合模型**：使用 Perceiver 风格跨图融合，对 RTL 与 ASM 上下文进行交互式编码。
- **多任务训练**：同时支持 RTL 边覆盖分类和图级覆盖率回归。
- **多覆盖率目标**：图级回归可选择 `branch`、`line`、`fsm`、`toggle`、`condition` 子集。
- **CodeBERT 指令编码**：可选启用 HuggingFace CodeBERT 为 ASM 指令生成文本编码；未启用时回退为零向量。
- **缓存与去重**：预处理阶段缓存 RTL/ASM 图结构与 ASM 编码，减少重复构建成本。
- **Joint Contrastive 训练**：支持 `(test_a, test_b, merged)` 三元组联合训练和超矩形对比损失。
- **Lightning 训练封装**：提供 checkpoint、early stopping、TensorBoard/CSV 日志和测试入口。

## 项目结构

```text
RTLMap/
├── main.py                         # 训练/测试 CLI 入口
├── pyproject.toml                  # Python 版本、依赖和 uv 配置
├── run_train.sh                    # 普通训练示例脚本
├── run_contrastive.sh              # joint contrastive 训练示例脚本
├── datasets/
│   ├── data_types.py               # DualGraphData、覆盖率目标定义
│   ├── datamodule.py               # 普通 manifest 数据集与 Lightning DataModule
│   ├── triple_datamodule.py        # 三元组对比学习数据管线
│   └── coverage_similarity.py      # 覆盖率相似度计算
├── cdfg_rtl/                       # RTL 图枚举和数据类型
├── cdfg_asm/                       # ASM 图枚举和数据类型
├── models/
│   ├── data_types.py               # ModelConfig、ModelOutput
│   ├── encoder.py                  # RTL/ASM 编码器
│   ├── interaction.py              # Perceiver 跨图融合
│   ├── model.py                    # DualGraphFusionModel 与任务头
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

Joint contrastive 模式使用 `contrastive_dataset.v1` 数据集。`manifest.json` 默认指向 `contrastive_samples.jsonlines`。

```json
{
  "schema_version": "contrastive_dataset.v1",
  "dataset_name": "contrastive_example",
  "samples": "contrastive_samples.jsonlines",
  "rtl_graphs": "rtl_graphs",
  "asm_graphs": "asm_graphs",
  "total": 1,
  "errors": 0
}
```

每行样本使用 `contrastive_sample.v1`，包含 `a`、`b` 和 `merged` 三路 targets：

```json
{
  "schema_version": "contrastive_sample.v1",
  "sample_id": "test_a::test_b::ALU",
  "module_name": "ALU",
  "rtl_graph": "rtl_graphs/ALU.json",
  "asm_a": "asm_graphs/test_a.json",
  "asm_b": null,
  "targets": {
    "a": {},
    "b": {},
    "merged": {}
  }
}
```

训练命令：

```bash
uv run python main.py \
    --dataset-dir /path/to/contrastive_dataset \
    --data-root ./data \
    --joint-contrastive \
    --use-hyperrectangle \
    --contrastive-batch-size 16 \
    --lambda-ce 1.0 \
    --lambda-cl 0.5
```

可选项：

- `--contrastive-triple-index`：指定三元组 jsonlines 文件；为空时读取 manifest 中的 `samples`。
- `--contrastive-loss-type`：`mse`、`bce` 或 `margin`。
- `--contrastive-margin`：`margin` loss 的不相似对交集上限。
- `--or-consistency-weight`：启用 merged 逻辑 OR 一致性约束，默认关闭。

## 常用 CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset-dir` | 必需 | 一个或多个 manifest 数据集目录 |
| `--data-root` | 必需 | 预处理缓存根目录 |
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
| `--joint-contrastive` | `false` | 启用三元组联合训练 |
| `--fast-dev-run` | `false` | Lightning 快速调试模式 |
| `--seed` | `None` | 全局随机种子 |

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
uv run python -m pytest -q tests/test_union_fusion.py
```

当前测试重点覆盖：

- manifest 数据集加载与稀疏标签处理
- 多 dataset 目录合并
- contrastive 三元组数据加载
- joint train DataModule 构建
- 多覆盖率 graph heads
- union fusion 的对称性与 merged 路径

## 缓存与输出

- `data_root/train/processed/`：普通训练样本缓存。
- `data_root/train/processed/rtl_*.pt`：去重后的 RTL 图结构缓存。
- `data_root/train/processed/asm_*.pt`：去重后的 ASM 图缓存。
- `data_root/asm_encoding_cache/`：CodeBERT pooled ASM 编码缓存。
- `checkpoint-dir/experiment-name/`：Lightning 日志和 checkpoint。

如果数据集内容或编码器配置变化，建议使用新的 `--data-root`，避免复用旧缓存导致实验不一致。

## 已知边界

- 本仓库当前 README 描述的是训练侧；原始 RTL/覆盖率报告提取工具不在当前代码树中作为主入口维护。
- 普通训练默认会在没有显式 val cache 时，从 train 数据集中按固定随机种子划分 9:1 验证集。
- `main.py` 暂未暴露所有 `TrainerConfig` 字段，高级训练配置建议使用 Python API。
- 启用 `--use-text-encoder` 需要可访问 HuggingFace 模型权重，离线环境需提前准备缓存。
