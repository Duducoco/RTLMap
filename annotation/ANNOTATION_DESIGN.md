# CDFG 覆盖率标注方案设计

## 1. 概述

本文档描述了 RTLMap 项目中覆盖率标注模块的设计方案。该模块的核心目标是：

1. **解析 VCS/URG 覆盖率报告**：从 HTML 格式的覆盖率报告中提取分支覆盖信息
2. **标注 CDFG 边**：将覆盖率信息映射到 CDFG（控制数据流图）的边上
3. **支持覆盖率传播**：通过数据流分析，将覆盖状态传播到相关边
4. **支持层次化模块**：当 CDFG 通过 Yosys flatten 后包含多个子模块的逻辑时，自动解析所有相关源文件的覆盖率

## 2. 数据结构

### 2.1 覆盖率数据结构

```python
class BranchType(Enum):
    """分支类型"""
    TERNARY = auto()  # 三元运算符 (a ? b : c)
    IF = auto()       # if-else 语句
    CASE = auto()     # case/unique case 语句

class BranchCoverage:
    """单个分支的覆盖率信息"""
    branch_type: BranchType      # 分支类型
    line_number: int             # 源码行号
    branch_index: int            # 分支索引（如 -1-, -2- 等）
    is_covered: bool             # 是否被覆盖
    condition_value: str         # 条件值（"0"=假分支，"1"=真分支）
    case_label: str              # case 标签（如 "RESET", "DECODE" 等）
    branch_path: str             # 完整分支路径描述
    source_file: str             # 源文件名（用于层次化模块匹配）
```

### 2.3 CDFG 边标注字段

```python
class Edge:
    """CDFG 边（已扩展覆盖率字段）"""
    source: str              # 源节点 ID
    target: str              # 目标节点 ID
    source_port: str         # 源端口名
    target_port: str         # 目标端口名
    edge_type: EdgeType      # 边类型 (DATA, CONTROL, CLOCK, RESET, ENABLE)

    # 覆盖率标注字段
    coverage_label: int      # 覆盖标签：-1=不适用, 0=未覆盖, 1=已覆盖
    coverage_type: str       # 覆盖类型："control", "data_true", "data_false", "propagated"
    branch_index: int        # 分支索引（if-else 链中的位置）
```

## 3. 层次化模块支持

### 3.1 问题背景

当使用 Yosys 进行 `flatten` 操作后，顶层模块的 CDFG 会包含所有子模块的逻辑。例如，`cv32e40p_core` 模块 flatten 后会包含来自以下源文件的逻辑：

- `cv32e40p_core.sv`
- `cv32e40p_controller.sv`
- `cv32e40p_decoder.sv`
- `cv32e40p_fifo.sv`
- `cv32e40p_sleep_unit.sv`
- ... 等等

每个 CDFG 节点的 `source_file` 属性记录了该节点对应的源文件，而覆盖率报告是按模块组织的。因此，标注时需要：

1. 收集 CDFG 中涉及的所有源文件
2. 为每个源文件找到对应的覆盖率报告
3. 根据节点的 `source_file` 和 `source_line` 进行精确匹配

## 4. 标注策略

### 4.1 MUX 节点标注

MUX（多路选择器）是 RTL 中控制流的核心，对应 if-else、三元运算符和 case 语句。

**MUX 端口结构：**
```
       A (数据0/假分支)  ─┐
                         ├─→ Y (输出)
       B (数据1/真分支)  ─┘
                  ↑
       S (选择信号) ──────
```

**标注逻辑：**

1. **控制边（S 端口）**：
   - `coverage_type = "control"`
   - `coverage_label = 1` 如果对应分支被覆盖，否则 `0`

2. **数据边（A/B 端口）**：
   - A 端口（假分支）：当 `condition_value = "0"` 的分支被覆盖时激活
     - `coverage_type = "data_false"`
   - B 端口（真分支）：当 `condition_value = "1"` 的分支被覆盖时激活
     - `coverage_type = "data_true"`

**示例：**

```systemverilog
// 源代码（行号 334-337）
if (fetch_enable_i == 1'b1)  // 行 334，分支标记 -2-
begin
    ctrl_fsm_ns = BOOT_SET;  // 行 336，真分支
end
// MISSING_ELSE             // 假分支
```

对应的覆盖率报告：
```
-2-  Status
 1   Covered      → B 边激活
 0   Covered      → A 边激活
```

生成的 CDFG MUX 节点标注：
```
Edge(S): coverage_label=1, coverage_type="control"
Edge(A): coverage_label=1, coverage_type="data_false"   # S=0 时选择
Edge(B): coverage_label=1, coverage_type="data_true"    # S=1 时选择
```

### 4.2 CASE 语句标注

CASE 语句在 Yosys 中被展开为嵌套的 MUX 或 PMUX。

**覆盖率报告格式：**
```
Branch                                           Status
(1.RESET )->(2)->(19.-)→..                      Covered
(1.RESET )->(!2)->(19.-)→..                     Covered
(1.BOOT_SET )->(3)->(19.-)→..                   Not Covered
(1.BOOT_SET )->(!3)->(19.-)→..                  Covered
```

**解析策略：**
1. 从 `branch_path` 中提取 case 标签（如 `RESET`, `BOOT_SET`）
2. 从决策点（如 `(2)`, `(!2)`）中提取条件值
3. 将覆盖状态映射到对应的 MUX 边

**精确范围匹配（`_collect_case_mux_nodes`）：**
1. 调用 `_find_case_range_from_source()` 从源码解析 `case`/`endcase` 范围
2. 处理嵌套的 case 语句（通过计数嵌套层级）
3. 收集范围内的所有 MUX 节点
4. 如果源码解析失败，回退到启发式方法

### 4.3 IF 语句标注

IF 语句在 Yosys 中被综合为 MUX 节点。对于 if-else-if 链，会生成串联的 MUX。

**精确范围匹配（`_collect_if_mux_nodes`）：**
1. 调用 `_find_if_range_from_source()` 从源码解析 if 语句范围
2. 使用正则表达式 `(?<!else\s)\bif\s*\(` 匹配 if 关键字（排除 else if）
3. 通过 `_find_if_end()` 方法确定 if 块的结束位置：
   - 检查是否有 `begin` 关键字
   - 如果没有 `begin`，认为是单行 if 或三元表达式
   - 如果有 `begin`，通过计数 `begin`/`end` 配对找到结束位置
4. 收集范围内的所有 MUX 节点
5. 如果源码解析失败，回退到启发式方法（连续 MUX 链收集）

**示例：**
```systemverilog
if (condition1) begin      // 行 100
    // true branch
end else if (condition2) begin  // 行 103
    // else-if branch
end else begin             // 行 106
    // else branch
end                        // 行 108
```

解析结果：`if_range = (100, 108)`

### 4.4 三元表达式标注

三元表达式 `cond ? true_val : false_val` 在 Yosys 中被综合为 MUX 节点。

**精确范围匹配（`_collect_ternary_mux_nodes`）：**
1. 调用 `_find_ternary_range_from_source()` 从源码解析三元表达式范围
2. 检查行中是否包含 `?` 和 `:` 操作符（排除注释中的）
3. 通过 `_find_ternary_end()` 方法确定表达式的结束位置
4. 收集范围内的所有 MUX 节点
5. 如果源码解析失败，回退到启发式方法

**嵌套三元表达式处理（`_find_ternary_end`）：**

`_find_ternary_end()` 方法使用以下策略处理嵌套三元表达式：

1. **跟踪括号深度**：
   - `paren_depth`：跟踪 `()` 的嵌套深度
   - `bracket_depth`：跟踪 `[]` 的嵌套深度

2. **区分三元表达式冒号和位选择冒号**：
   - 位选择的冒号在 `[]` 内部，如 `data[7:0]`
   - 三元表达式的冒号在 `[]` 外部
   - 只有当 `bracket_depth == 0` 时，`:` 才被视为三元表达式的冒号

3. **跟踪三元表达式嵌套深度**：
   - `ternary_depth`：未匹配的 `?` 数量
   - 遇到 `?` 时增加 `ternary_depth`
   - 遇到三元表达式的 `:` 时减少 `ternary_depth`
   - 当 `ternary_depth == 0` 且括号平衡时，表达式结束

4. **处理字符串字面量**：
   - 跳过引号内的内容，避免误匹配

5. **结束条件**：
   - 分号 `;` 表示语句结束
   - 逗号 `,` 在所有三元表达式匹配后表示表达式结束
   - 所有 `?` 都找到匹配的 `:` 且括号平衡

**嵌套三元表达式示例：**

```systemverilog
// 示例 1：简单嵌套
assign out = a ? b : (c ? d : e);

// 示例 2：多层嵌套
assign out = a ? (b ? c : d) : (e ? f : g);

// 示例 3：跨行嵌套
assign result = sel1 ? val1 :    // 行 50
                sel2 ? val2 :    // 行 51
                       val3;     // 行 52

// 示例 4：带位选择的三元表达式
assign out = sel ? data1[7:0] : data2[15:8];  // [7:0] 和 [15:8] 中的 : 不会被误识别
```

解析结果：
- 示例 1：`ternary_range = (行号, 行号)`，收集 2 个 MUX
- 示例 2：`ternary_range = (行号, 行号)`，收集 3 个 MUX
- 示例 3：`ternary_range = (50, 52)`，收集 2 个 MUX
- 示例 4：`ternary_range = (行号, 行号)`，收集 1 个 MUX

### 4.5 行号偏移处理机制

VCS 覆盖率报告中的行号与 Yosys JSON 中的行号可能存在偏移，这是由于 include 文件、宏展开等处理方式不同导致的。偏移量可达 30+ 行。

**偏移量学习机制：**

1. **配置常量**：
   ```python
   DEFAULT_OFFSET_SEARCH_RANGE = (-5, 40)  # 默认搜索范围
   LEARNED_OFFSET_TOLERANCE = 5  # 学习到偏移量后的容差范围
   ```

2. **`_get_offset_search_order()` 方法**：
   - 如果已学习到偏移量，优先搜索该偏移量附近的范围
   - 搜索顺序：学习到的偏移量 → 容差范围内的偏移量 → 其他偏移量
   - 如果未学习到偏移量，优先尝试 0 偏移，然后按距离递增

3. **`_update_learned_offset()` 方法**：
   - 首次学习时直接设置偏移量
   - 已有偏移量时验证新偏移量是否在容差范围内
   - 如果差异过大，输出警告信息

**搜索顺序示例**：

假设已学习到偏移量为 10，容差为 5，搜索范围为 (-5, 20)：
1. 优先级1：`[10]`（学习到的偏移量）
2. 优先级2：`[9, 11, 8, 12, 7, 13, 6, 14, 5, 15]`（容差范围内）
3. 优先级3：`[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 16, 17, 18, 19, 20]`（其他）

**应用场景**：

- `_find_case_range_from_source()`：CASE 语句范围查找
- `_find_if_range_from_source()`：IF 语句范围查找
- `_find_ternary_range_from_source()`：三元表达式范围查找

### 4.6 时序节点标注

时序节点（DFF、ADFF 等）的数据输入边也可以基于行覆盖率进行标注。

```python
# 时序节点端口
D   -> 数据输入
Q   -> 数据输出
CLK -> 时钟
ARST/SRST -> 复位
```

标注策略：
- D 端口边：基于该行的覆盖状态标注
