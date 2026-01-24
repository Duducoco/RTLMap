# RTLMap

RTLMap 是一个用于从 Yosys RTLIL JSON 提取控制数据流图 (CDFG) 并用 VCS/URG 覆盖率数据进行标注的工具。标注后的图可用于基于 GNN 的硬件验证训练数据。

## 功能特性

- **CDFG 提取**：从 Yosys 综合后的 RTLIL JSON 中提取控制数据流图
- **覆盖率标注**：解析 VCS/URG HTML 覆盖率报告，将分支覆盖信息映射到 CDFG 边
- **可视化输出**：生成带覆盖率着色的 SVG 图形（绿色=已覆盖，红色=未覆盖）
- **ASM CDFG 提取**：从 RISC-V 汇编代码提取基本块级 CDFG，支持 RV32I/M/F/C 和 Xpulp 扩展
- **双图融合 GNN**：RTL-Assembly 双图交互式编码，用于覆盖率预测
- **PyTorch Lightning 训练**：多任务训练框架（边分类 + 图回归），支持 TensorBoard
- **自动综合**：自动调用 Yosys（需要 oss-cad-suite）生成 RTLIL JSON
- **智能缓存**：自动缓存源文件到 HTML 的映射关系，加速重复运行

## 数据流管线

```
RTL (SystemVerilog)                      Assembly (.S)
        │ (YosysRunner + slang plugin)          │ (asm_cdfg.AsmParser)
        ▼                                       ▼
design.json (RTLIL JSON)                 Instruction list
        │ (cdfg.CDFGExtractor)                  │ (asm_cdfg.BasicBlockBuilder)
        ▼                                       ▼
CDFG object (nodes + edges)              BasicBlock list
        │ (annotation.CoverageParser)           │ (asm_cdfg.AsmCDFGExtractor)
        ▼                                       ▼
coverage.html -> BranchCoverage data     ASM CDFG (基本块级图)
        │ (annotation.CoverageAnnotator)        │
        ▼                                       │
Annotated CDFG (edges with coverage_label)      │
        │ (cdfg.CDFGExporter)                   │
        ▼                                       │
cdfg_annotated.json / cdfg_output.svg           │
        │                                       │
        └───────────────┬───────────────────────┘
                        │ (models.DualGraphData)
                        ▼
            DualGraphFusionModel (双图交互编码)
                        │ (trainer.train_model)
                        ▼
            覆盖率预测模型 (边分类 + 图回归)
```

## 安装

### 依赖

- Python >= 3.13
- [uv](https://github.com/astral-sh/uv) (推荐的 Python 包管理器)
- [oss-cad-suite](https://github.com/YosysHQ/oss-cad-suite-build) (用于 Yosys 综合)
- Graphviz (用于 SVG 可视化)
- PyTorch 2.6+ with CUDA 12.4 (用于 GNN 训练)
- PyTorch Geometric 2.7+ (图神经网络库)
- PyTorch Lightning 2.6+ (训练框架)

### 安装步骤

```bash
# 克隆仓库
git clone <repository-url>
cd RTLMap

# 使用 uv 安装依赖
uv sync

# 验证安装
uv run python data_annotate.py --help
```

### 环境配置

确保 oss-cad-suite 已安装并可用。YosysRunner 会自动设置以下环境变量：
- `YOSYSHQ_ROOT`
- `PATH`
- `SSL_CERT_FILE`

## 使用方法

### 快速开始

```bash
# 基本 CDFG 提取与覆盖率标注
uv run python data_annotate.py \
    --design_name cv32e40p \
    --module_name cv32e40p_core \
    -c designs/cv32e40p/coverage_reports/coverage_report1
```

### 完整示例

```bash
# 1. 仅提取 CDFG（不标注覆盖率）
uv run python data_annotate.py \
    --design_name cv32e40p \
    --module_name cv32e40p_core \
    --no-coverage

# 2. 提取并标注覆盖率
uv run python data_annotate.py \
    --design_name cv32e40p \
    --module_name cv32e40p_core \
    -c designs/cv32e40p/coverage_reports/coverage_report1

# 3. 启用覆盖率传播（标注非分支边）
uv run python data_annotate.py \
    --design_name cv32e40p \
    --module_name cv32e40p_core \
    -c designs/cv32e40p/coverage_reports/coverage_report1 \
    --propagate

# 4. 生成 SVG 可视化
uv run python data_annotate.py \
    --design_name cv32e40p \
    --module_name cv32e40p_core \
    -c designs/cv32e40p/coverage_reports/coverage_report1 \
    --svg

# 5. 完整流程（标注 + 传播 + 可视化 + 详细输出）
uv run python data_annotate.py \
    --design_name cv32e40p \
    --module_name cv32e40p_core \
    -c designs/cv32e40p/coverage_reports/coverage_report1 \
    --propagate --svg -v
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--design_name` | 设计名称（对应 `designs/` 下的文件夹名） | **必需** |
| `--module_name` | 顶层模块名称 | **必需** |
| `-c, --coverage-dir` | 覆盖率报告目录 | - |
| `-o, --output` | 输出 JSON 文件路径 | `cdfg_annotated.json` |
| `-i, --instance` | 覆盖率实例标签 | 使用最后一个实例 |
| `--no-coverage` | 仅提取 CDFG，不标注覆盖率 | `false` |
| `--propagate` | 启用覆盖率传播到非分支边 | `false` |
| `--svg` | 生成 SVG 可视化图 | `false` |
| `--svg-output` | SVG 输出文件路径 | `cdfg_output.svg` |
| `-v, --verbose` | 显示详细信息 | `false` |

### 直接使用 YosysRunner

```bash
# 从文件列表生成 RTLIL JSON
uv run python tools/YosysRunner.py \
    --top <module_name> \
    --flist <manifest.flist> \
    --output-dir json/

# 从指定文件生成
uv run python tools/YosysRunner.py \
    --top <module_name> \
    --files file1.sv file2.sv \
    --output-dir json/
```

### ASM CDFG 提取

```bash
# 从 RISC-V 汇编文件提取基本块级 CDFG
uv run python -m asm_cdfg.extractor input.S -o output.json -v
```

### 运行测试

```bash
# 模型单元测试
uv run python -m models.test_model

# 训练器单元测试
uv run python trainer/test_trainer.py
```

## 项目结构

```
RTLMap/
├── data_annotate.py          # 主入口脚本
├── pyproject.toml            # 项目配置
├── CLAUDE.md                 # Claude Code 指导文档
├── README.md                 # 项目说明
│
├── cdfg/                     # CDFG 提取与可视化模块
│   ├── __init__.py
│   ├── types.py              # 节点/边类型定义 (Node, Edge, CDFG)
│   ├── extractor.py          # RTLIL JSON -> CDFG 提取
│   ├── classifier.py         # Yosys cell 类型分类
│   ├── exporter.py           # 导出 JSON/Graphviz
│   ├── visualizer.py         # SVG 可视化
│   └── analyzer.py           # 图分析工具
│
├── annotation/               # 覆盖率标注模块
│   ├── __init__.py
│   ├── parser.py             # VCS/URG HTML 解析
│   └── annotator.py          # 覆盖率 -> CDFG 边标注
│
├── asm_cdfg/                 # RISC-V 汇编 CDFG 模块
│   ├── __init__.py
│   ├── asm_types.py          # 指令和基本块类型定义
│   ├── parser.py             # 汇编文件解析器
│   ├── classifier.py         # 指令分类器 (RV32I/M/F/C, Xpulp)
│   ├── bb_builder.py         # 基本块构建器
│   ├── extractor.py          # 基本块 -> CDFG 提取
│   └── visualizer.py         # ASM CDFG 可视化
│
├── models/                   # 双图融合 GNN 模型
│   ├── __init__.py
│   ├── data_types.py         # 数据类型 (DualGraphData, ModelConfig)
│   ├── encoder.py            # 图编码器 (GINEConv + 跨图交互)
│   ├── interaction.py        # 跨图注意力机制
│   ├── model.py              # 完整模型 + 任务头
│   └── test_model.py         # 模型单元测试
│
├── trainer/                  # PyTorch Lightning 训练框架
│   ├── __init__.py
│   ├── config.py             # 训练超参数配置
│   ├── callbacks.py          # Callback 工厂类
│   ├── datamodule.py         # 数据模块封装
│   ├── module.py             # Lightning 模型封装
│   ├── utils.py              # 便捷训练函数
│   └── test_trainer.py       # 训练器单元测试
│
├── tools/                    # 工具脚本
│   ├── __init__.py
│   └── YosysRunner.py        # Yosys 执行封装
│
└── designs/                  # 设计文件目录
    ├── cv32e40p/             # CV32E40P RISC-V 处理器
    └── openc910/             # OpenC910 处理器
```

## 设计文件结构

每个设计需要按以下结构组织：

```
designs/<design_name>/
├── <design_name>.flist          # Verilog/SystemVerilog 文件列表
├── source/                      # RTL 源文件目录
│   └── rtl/
│       └── *.sv
├── RTLIL_json/                  # 自动生成的 RTLIL JSON
│   └── <module_name>.json
├── module2html.json             # 模块名 -> 覆盖率 HTML 文件映射
├── source2html.json             # 源文件名 -> 覆盖率 HTML 映射（自动生成）
└── coverage_reports/
    └── <report_name>/           # VCS/URG 生成的覆盖率报告
        └── *.html
```

### module2html.json 格式

```json
{
    "cv32e40p_core": "mod1.html",
    "cv32e40p_controller": "mod2.html",
    "cv32e40p_decoder": "mod3.html"
}
```

## 输出格式

### 标注后的 JSON

```json
{
    "module_name": "cv32e40p_int_controller",
    "nodes": [
        {
            "id": "$procmux$1614",
            "node_type": "MUX",
            "cell_type": "$mux",
            "source_file": "cv32e40p_int_controller.sv",
            "source_line": 97,
            "width": 32,
            "input_ports": ["A", "B", "S"],
            "output_ports": ["Y"]
        }
    ],
    "edges": [
        {
            "source": "$4",
            "target": "$procmux$1614",
            "source_port": "Y",
            "target_port": "S",
            "edge_type": "CONTROL",
            "coverage_label": 1,
            "coverage_type": "control",
            "branch_index": 0
        }
    ]
}
```

### 覆盖标签说明

| coverage_label | 含义 |
|----------------|------|
| -1 | 未标注（无对应覆盖率数据） |
| 0 | 未覆盖 |
| 1 | 已覆盖 |

| coverage_type | 含义 |
|---------------|------|
| control | 控制边（MUX 的 S 端口） |
| data_true | 真分支数据边（MUX 的 B 端口） |
| data_false | 假分支数据边（MUX 的 A 端口） |
| propagated | 传播标注边 |
| always | 必然执行边（输入/常量） |

## 核心模块说明

### CDFG 节点类型

| NodeType | 说明 | 对应 Yosys Cell |
|----------|------|-----------------|
| INPUT | 输入端口 | - |
| OUTPUT | 输出端口 | - |
| CONSTANT | 常量 | - |
| MUX | 多路选择器 | `$mux`, `$pmux` |
| SEQUENTIAL | 时序元件 | `$dff`, `$aldff`, `$adff` 等 |
| LOGIC | 逻辑运算 | `$and`, `$or`, `$not` 等 |
| ARITHMETIC | 算术运算 | `$add`, `$sub`, `$mul` 等 |
| COMPARE | 比较运算 | `$eq`, `$lt`, `$gt` 等 |
| SHIFT | 移位运算 | `$shl`, `$shr` 等 |

### CDFG 边类型

| EdgeType | 说明 |
|----------|------|
| DATA | 数据流边 |
| CONTROL | 控制流边（MUX 选择信号） |
| CLOCK | 时钟信号边 |
| RESET | 复位信号边 |
| ENABLE | 使能信号边 |

### 覆盖率标注

标注器支持三种分支类型的自动检测和标注：

1. **IF 语句**
   - 支持 `if-else` 和 `if-else-if` 链
   - 支持带或不带 `begin`/`end` 块
   - 自动处理优先编码器风格的级联 if-else

2. **CASE 语句**
   - 支持 `case`/`casex`/`casez`
   - 支持 `unique case`
   - 自动解析 `case`/`endcase` 边界

3. **三元表达式**
   - 支持 `cond ? true_val : false_val`
   - 支持嵌套三元表达式

## 示例输出

```
正在从 designs/cv32e40p/RTLIL_json/cv32e40p_int_controller.json 提取 CDFG...
模块: cv32e40p_int_controller
节点数: 54
边数: 149

CDFG 中包含 1 个源文件的节点
匹配到 1 个覆盖率报告文件

--- 正在处理: mod36.html ---
  模块: cv32e40p_int_controller
  源文件: cv32e40p_int_controller.sv
  实例: inst_tag_59
  从源码解析到 if 范围: 97-148
行 96: 找到 32 个 MUX 节点，对应 33 个分支
  标注: 32 MUX, 136 边

正在传播覆盖率...

==================================================
总体标注统计:
  处理文件数: 1
  MUX 节点: 32/32 (直接匹配覆盖率分支)
  已标注边: 149/149
  已覆盖边: 55
  未覆盖边: 94

边覆盖率: 36.91%

已导出 CDFG: cdfg_annotated.json
已导出 SVG: cdfg_output.svg
```

## 故障排查

### 常见问题

1. **所有边都显示为已覆盖 (100%)**
   - 原因：行号匹配不正确
   - 解决：使用 `-v` 查看详细输出，检查警告信息
   - 标注器内置验证：MUX 数量超过预期 3 倍时会报警

2. **If-else-if 链未被检测**
   - 原因：源文件不符合预期模式
   - 解决：检查是否为不带 `begin`/`end` 的优先编码器风格

3. **覆盖率报告中找不到模块**
   - 原因：`module2html.json` 映射不正确
   - 解决：检查映射文件或重新生成

4. **Yosys 综合失败**
   - 原因：oss-cad-suite 未正确安装
   - 解决：确保 Yosys 和 slang 插件可用

5. **SVG 生成失败**
   - 原因：Graphviz 未安装
   - 解决：安装 Graphviz (`apt install graphviz` 或 `brew install graphviz`)

## 开发指南

### 添加新设计

1. 在 `designs/` 下创建设计目录：
   ```bash
   mkdir -p designs/my_design/{source/rtl,coverage_reports,RTLIL_json}
   ```

2. 创建文件列表 `my_design.flist`：
   ```
   source/rtl/module1.sv
   source/rtl/module2.sv
   ```

3. 将 VCS/URG 覆盖率报告放入 `coverage_reports/report1/`

4. 运行标注：
   ```bash
   uv run python data_annotate.py \
       --design_name my_design \
       --module_name top_module \
       -c designs/my_design/coverage_reports/report1
   ```

### API 使用

```python
from cdfg import CDFGExtractor, CDFGExporter, CDFG
from annotation import CoverageParser, annotate_cdfg_with_coverage

# 提取 CDFG
extractor = CDFGExtractor("design.json")
cdfg = extractor.extract()

# 标注覆盖率
stats = annotate_cdfg_with_coverage(
    cdfg,
    "coverage.html",
    instance_tag="inst_tag_1",
    propagate=True
)

# 导出
exporter = CDFGExporter(cdfg)
exporter.to_json("output.json")
exporter.to_graphviz("output.svg", show_coverage=True)
```

### GNN 训练 API

```python
from models import create_model, ModelConfig, DualGraphData
from trainer import train_model, TrainerConfig

# 创建模型配置
model_config = ModelConfig(
    rtl_node_dim=64,
    asm_node_dim=64,
    hidden_dim=256,
    num_gnn_layers=4
)

# 创建训练配置
trainer_config = TrainerConfig(
    learning_rate=1e-4,
    max_epochs=100,
    batch_size=32
)

# 准备数据（DualGraphData 包含 RTL 和 ASM 双图）
# train_data, val_data = ...

# 训练模型
module, trainer = train_model(
    model_config=model_config,
    trainer_config=trainer_config,
    train_data=train_data,
    val_data=val_data,
    experiment_name="coverage_prediction"
)
```

### ASM CDFG API

```python
from asm_cdfg import AsmCDFGExtractor

# 从汇编文件提取 CDFG
extractor = AsmCDFGExtractor(verbose=True)
cdfg = extractor.extract_from_file("test.S")

print(f"节点数: {len(cdfg.nodes)}")
print(f"边数: {len(cdfg.edges)}")
```

### 开发环境设置

```bash
# 克隆并安装开发依赖
git clone <repository-url>
cd RTLMap
uv sync

# 运行测试（如果有）
uv run pytest
```

### 代码风格

- 使用中文注释和日志输出
- 遵循 Python 类型注解规范
- 使用 dataclass 定义数据结构
- 函数和类需要完整的 docstring
