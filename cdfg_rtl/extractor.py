#!/usr/bin/env python3
"""
CDFG 提取器 - 从 RTLIL JSON 提取控制数据流图
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict

from .data_types import Node, Edge, CDFG, NodeType, EdgeType
from .classifier import CellClassifier


class CDFGExtractor:
    """CDFG 提取器"""

    def __init__(self, json_data: Union[str, Path, dict]):
        """
        初始化提取器

        Args:
            json_data: JSON 文件路径（字符串或 Path 对象）或已解析的字典
        """
        if isinstance(json_data, (str, Path)):
            with open(json_data, "r", encoding="utf-8") as f:
                self.design = json.load(f)
        else:
            self.design = json_data

        self.classifier = CellClassifier()
        self.cdfg_cache: Dict[str, CDFG] = {}

    def extract(self, module_name: Optional[str] = None) -> CDFG:
        """
        提取指定模块的 CDFG

        Args:
            module_name: 模块名，None 则提取第一个模块

        Returns:
            CDFG 对象
        """
        modules = self.design.get("modules", {})

        if not modules:
            raise ValueError("JSON 中没有找到模块定义")

        if module_name is None:
            module_name = next(iter(modules.keys()))

        if module_name not in modules:
            raise ValueError(f"模块 '{module_name}' 不存在")

        if module_name in self.cdfg_cache:
            return self.cdfg_cache[module_name]

        module = modules[module_name]
        cdfg = CDFG(module_name=module_name)

        # 步骤 1: 提取端口节点
        self._extract_ports(module, cdfg)

        # 步骤 2: 提取常数节点
        self._extract_constants(module, cdfg)

        # 步骤 3: 提取 cell 节点
        self._extract_cells(module, cdfg)

        # 步骤 4: 建立连接关系
        self._build_connections(module, cdfg)

        # 步骤 5: 提取边
        self._extract_edges(module, cdfg)

        self.cdfg_cache[module_name] = cdfg
        return cdfg

    def _extract_ports(self, module: dict, cdfg: CDFG):
        """提取端口节点"""
        ports = module.get("ports", {})

        for port_name, port_info in ports.items():
            direction = port_info.get("direction", "input")
            bits = port_info.get("bits", [])

            node_type = NodeType.INPUT if direction == "input" else NodeType.OUTPUT
            node_id = f"port_{port_name}"

            node = Node(
                id=node_id,
                name=port_name,
                node_type=node_type,
                cell_type=direction,
                width=len(bits),
                input_ports=["in"] if direction == "output" else [],
                output_ports=["out"] if direction == "input" else [],
            )
            cdfg.nodes[node_id] = node

            # 记录 bit 驱动关系
            if direction == "input":
                for bit in bits:
                    if isinstance(bit, int):
                        cdfg.bit_to_driver[bit] = (node_id, "out")

    def _extract_constants(self, module: dict, cdfg: CDFG):
        """提取常数节点"""
        # 常数节点在处理 cell 连接时动态创建
        pass

    def _extract_cells(self, module: dict, cdfg: CDFG):
        """提取 cell 节点"""
        cells = module.get("cells", {})

        for cell_name, cell_info in cells.items():
            cell_type = cell_info.get("type", "")
            parameters = self._parse_parameters(cell_info.get("parameters", {}))
            attributes = cell_info.get("attributes", {})
            port_directions = cell_info.get("port_directions", {})
            connections = cell_info.get("connections", {})

            # 确定节点类型
            node_type = self.classifier.classify(cell_type)

            # 确定位宽
            width = parameters.get("WIDTH", parameters.get("Y_WIDTH", 1))

            # 解析源代码位置
            src_attr = attributes.get("src", "")
            source_file, source_line, stmt_start_line = (
                CellClassifier.parse_source_location(src_attr)
            )

            # 提取输入输出端口
            input_ports = []
            output_ports = []

            for port_name in connections.keys():
                if port_directions.get(
                    port_name
                ) == "output" or self.classifier.is_output_port(cell_type, port_name):
                    output_ports.append(port_name)
                else:
                    input_ports.append(port_name)

            node = Node(
                id=cell_name,
                name=self._get_display_name(cell_name),
                node_type=node_type,
                cell_type=cell_type,
                width=width,
                parameters=parameters,
                attributes=attributes,
                input_ports=input_ports,
                output_ports=output_ports,
                source_line=source_line,
                source_file=source_file,
                stmt_start_line=stmt_start_line,
            )
            cdfg.nodes[cell_name] = node

    def _build_connections(self, module: dict, cdfg: CDFG):
        """建立 bit 到节点的映射关系"""
        cells = module.get("cells", {})

        for cell_name, cell_info in cells.items():
            cell_type = cell_info.get("type", "")
            connections = cell_info.get("connections", {})
            port_directions = cell_info.get("port_directions", {})

            for port_name, bits in connections.items():
                is_output = port_directions.get(
                    port_name
                ) == "output" or self.classifier.is_output_port(cell_type, port_name)

                for bit in bits:
                    if isinstance(bit, int):
                        if is_output:
                            cdfg.bit_to_driver[bit] = (cell_name, port_name)
                        else:
                            cdfg.bit_to_consumers[bit].append((cell_name, port_name))

    def _extract_edges(self, module: dict, cdfg: CDFG):
        """提取所有边"""
        cells = module.get("cells", {})
        ports = module.get("ports", {})

        # 用于追踪常数节点
        const_nodes: Dict[str, str] = {}  # 常数值 -> 节点ID

        # 处理 cell 连接
        for cell_name, cell_info in cells.items():
            cell_type = cell_info.get("type", "")
            connections = cell_info.get("connections", {})
            port_directions = cell_info.get("port_directions", {})

            for port_name, bits in connections.items():
                is_output = port_directions.get(
                    port_name
                ) == "output" or self.classifier.is_output_port(cell_type, port_name)

                if is_output:
                    continue  # 输出端口不需要找驱动

                # 对于输入端口，找到驱动源
                edge_type = self.classifier.get_edge_type(port_name, cell_type)

                # 按连续的驱动源分组
                source_groups = self._group_bits_by_driver(bits, cdfg, const_nodes)

                for (source_id, source_port), bit_list in source_groups.items():
                    edge = Edge(
                        source=source_id,
                        target=cell_name,
                        source_port=source_port,
                        target_port=port_name,
                        edge_type=edge_type,
                        bits=bit_list,
                        width=len(bit_list),
                    )
                    cdfg.edges.append(edge)

        # 处理输出端口连接
        for port_name, port_info in ports.items():
            if port_info.get("direction") != "output":
                continue

            bits = port_info.get("bits", [])
            port_node_id = f"port_{port_name}"

            source_groups = self._group_bits_by_driver(bits, cdfg, const_nodes)

            for (source_id, source_port), bit_list in source_groups.items():
                edge = Edge(
                    source=source_id,
                    target=port_node_id,
                    source_port=source_port,
                    target_port="in",
                    edge_type=EdgeType.DATA,
                    bits=bit_list,
                    width=len(bit_list),
                )
                cdfg.edges.append(edge)

        # 添加常数节点到 CDFG
        for const_value, node_id in const_nodes.items():
            node = Node(
                id=node_id,
                name=const_value,
                node_type=NodeType.CONSTANT,
                cell_type="constant",
                width=1,
                output_ports=["out"],
            )
            cdfg.nodes[node_id] = node

    def _group_bits_by_driver(
        self, bits: List, cdfg: CDFG, const_nodes: Dict[str, str]
    ) -> Dict[Tuple[str, str], List[int]]:
        """将 bits 按驱动源分组"""
        groups = defaultdict(list)

        for i, bit in enumerate(bits):
            if isinstance(bit, str):
                # 常数
                if bit not in const_nodes:
                    const_id = f"const_{bit}_{len(const_nodes)}"
                    const_nodes[bit] = const_id
                source = (const_nodes[bit], "out")
            elif isinstance(bit, int):
                if bit in cdfg.bit_to_driver:
                    source = cdfg.bit_to_driver[bit]
                else:
                    # 未知驱动（可能是未连接）
                    source = ("unconnected", "out")
            else:
                continue

            groups[source].append(bit if isinstance(bit, int) else i)

        return groups

    def _parse_parameters(self, params: dict) -> dict:
        """解析参数（将二进制字符串转为整数）"""
        result = {}
        for key, value in params.items():
            if isinstance(value, str) and all(c in "01" for c in value):
                result[key] = int(value, 2)
            else:
                result[key] = value
        return result

    def _get_display_name(self, cell_name: str) -> str:
        """获取显示名称"""
        # 简化自动生成的名称
        if "$" in cell_name:
            parts = cell_name.split("$")
            if len(parts) >= 2:
                return f"${parts[1]}"
        return cell_name

    def extract_all(self) -> Dict[str, CDFG]:
        """提取所有模块的 CDFG"""
        result = {}
        for module_name in self.design.get("modules", {}).keys():
            result[module_name] = self.extract(module_name)
        return result
