# RTLMap

RTLMap 是一个用于从 Yosys RTLIL JSON 提取控制数据流图 (CDFG) 并用 VCS/URG 覆盖率数据进行标注的工具。标注后的图可用于基于 GNN 的硬件验证训练数据。

## 功能特性

- **CDFG 提取**：从 Yosys 综合后的 RTLIL JSON 中提取控制数据流图
- **覆盖率标注**：解析 VCS/URG HTML 覆盖率报告，将分支覆盖信息映射到 CDFG 边
- **可视化输出**：生成带覆盖率着色的 SVG 图形（绿色=已覆盖，红色=未覆盖）
- **层次化支持**：支持 flatten 后包含多个子模块的设计
- **自动综合**：自动调用 Yosys（需要 oss-cad-suite）生成 RTLIL JSON

## 数据流管线

```
RTL (SystemVerilog)
        | (YosysRunner + slang plugin)
        v
design.json (RTLIL JSON)
        | (cdfg.CDFGExtractor)
        v
CDFG object (nodes + edges)
        | (annotation.CoverageParser)
        v
coverage.html -> BranchCoverage data
        | (annotation.CoverageAnnotator)
        v
Annotated CDFG (edges with coverage_label)
        | (cdfg.CDFGExporter)
        v
cdfg_annotated.json / cdfg_output.svg
```

## 安装

### 依赖

- Python >= 3.13
- [uv](https://github.com/astral-sh/uv) (推荐的 Python 包管理器)
- [oss-cad-suite](https://github.com/YosysHQ/oss-cad-suite-build) (用于 Yosys 综合)
- Graphviz (用于 SVG 可视化)

### 安装步骤

```bash
# 克隆仓库
git clone <repository-url>
cd RTLMap

# 使用 uv 安装依赖
uv sync
```

## 使用方法

### 基本用法

```bash
# 基本 CDFG 提取与覆盖率标注
uv run python data_annotate.py \
    --design_name cv32e40p \
    --module_name cv32e40p_core \
    -c designs/cv32e40p/coverage_reports/coverage_report1

# 仅提取 CDFG（不标注覆盖率）
uv run python data_annotate.py \
    --design_name cv32e40p \
    --module_name cv32e40p_core \
    --no-coverage

# 生成 SVG 可视化
uv run python data_annotate.py \
    --design_name cv32e40p \
    --module_name cv32e40p_core \
    -c designs/cv32e40p/coverage_reports/coverage_report1 \
    --svg

# 启用覆盖率传播（标注非分支边）
uv run python data_annotate.py \
    --design_name cv32e40p \
    --module_name cv32e40p_core \
    -c designs/cv32e40p/coverage_reports/coverage_report1 \
    --propagate

# 显示详细信息
uv run python data_annotate.py \
    --design_name cv32e40p \
    --module_name cv32e40p_core \
    -c designs/cv32e40p/coverage_reports/coverage_report1 \
    -v
```

### 命令行参数

| 参数 | 说明 |
|------|------|
| `--design_name` | 设计名称（对应 `designs/` 下的文件夹名） |
| `--module_name` | 顶层模块名称 |
| `-c, --coverage-dir` | 覆盖率报告目录 |
| `-o, --output` | 输出 JSON 文件路径（默认: `cdfg_annotated.json`） |
| `-i, --instance` | 覆盖率实例标签（默认: 使用最后一个实例） |
| `--no-coverage` | 仅提取 CDFG，不标注覆盖率 |
| `--propagate` | 启用覆盖率传播到非分支边 |
| `--svg` | 生成 SVG 可视化图 |
| `--svg-output` | SVG 输出文件路径（默认: `cdfg_output.svg`） |
| `-v, --verbose` | 显示详细信息 |

### 直接使用 YosysRunner

```bash
# 从文件列表生成 RTLIL JSON
uv run python tools/YosysRunner.py \
    --top <module_name> \
    --flist <manifest.flist> \
    --output-dir json/

# 从文件列表生成
uv run python tools/YosysRunner.py \
    --top <module_name> \
    --files file1.sv file2.sv \
    --output-dir json/
```

## 项目结构

```
RTLMap/
├── data_annotate.py          # 主入口脚本
├── cdfg/                     # CDFG 提取与可视化模块
│   ├── __init__.py
│   ├── extractor.py          # RTLIL JSON -> CDFG 提取
│   ├── types.py              # 节点/边类型定义
│   ├── classifier.py         # Yosys cell 类型分类
│   ├── exporter.py           # 导出 JSON/Graphviz
│   ├── visualizer.py         # SVG 可视化
│   └── analyzer.py           # 图分析工具
├── annotation/               # 覆盖率标注模块
│   ├── __init__.py
│   ├── parser.py             # VCS/URG HTML 解析
│   ├── annotator.py          # 覆盖率 -> CDFG 边标注
│   └── ANNOTATION_DESIGN.md  # 标注方案设计文档
├── tools/                    # 工具脚本
│   ├── __init__.py
│   └── YosysRunner.py        # Yosys 执行封装
└── designs/                  # 设计文件目录
    └── <design_name>/
        ├── <design_name>.flist
        ├── RTLIL_json/
        ├── module2html.json
        └── coverage_reports/
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
            "source_file": "cv32e40p_int_controller.sv",
            "source_line": 97
        }
    ],
    "edges": [
        {
            "source": "$4",
            "target": "$procmux$1614",
            "source_port": "Y",
            "target_port": "S",
            "edge_type": "CONTROL",
            "coverage_label": 0,
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

### 覆盖率标注

标注器支持三种分支类型：

1. **IF 语句**：包括 `if-else` 和 `if-else-if` 链（带或不带 `begin`/`end`）
2. **CASE 语句**：`case`/`casex`/`casez` 和 `unique case`
3. **三元表达式**：`cond ? true_val : false_val`，支持嵌套

详细设计文档请参阅 [annotation/ANNOTATION_DESIGN.md](annotation/ANNOTATION_DESIGN.md)

## 示例输出

```
正在从 designs/cv32e40p/RTLIL_json/cv32e40p_int_controller.json 提取 CDFG...
模块: cv32e40p_int_controller
节点数: 54
边数: 149

找到 1 个覆盖率报告文件

--- 正在处理: mod36.html ---
  模块: cv32e40p_int_controller
  源文件: cv32e40p_int_controller.sv
  实例: inst_tag_59
  从源码解析到 if 范围: 97-148
行 96: 找到 32 个 MUX 节点，对应 33 个分支
  标注: 32 MUX, 136 边

==================================================
总体标注统计:
  处理文件数: 1
  MUX 节点: 32/32 (直接匹配覆盖率分支)
  已标注边: 134/149
  已覆盖边: 40
  未覆盖边: 94

边覆盖率: 29.85%

已导出 CDFG: cdfg_annotated.json
已导出 SVG: cdfg_output.svg
```

## 许可证

[待添加]

## 贡献

欢迎提交 Issue 和 Pull Request！
