#!/usr/bin/env python3
"""
RISC-V 汇编解析器

解析 .S 汇编文件，生成 Instruction 列表。
支持各种操作数格式和伪指令。
"""

import re
from pathlib import Path
from typing import List, Optional, Dict, Union

from .data_types import (
    Instruction,
    InstrFormat,
    InstrCategory,
    REGISTER_ABI_TO_X,
    FLOAT_REGISTER_ABI_TO_F,
)
from .classifier import InstructionClassifier


class AsmParser:
    """RISC-V 汇编解析器"""

    # 寄存器名模式
    REG_PATTERN = re.compile(
        r"^(x\d+|f\d+|zero|ra|sp|gp|tp|t[0-6]|s[0-9]|s1[01]|a[0-7]|"
        r"fp|ft\d+|fs\d+|fa\d+)$",
        re.IGNORECASE,
    )

    # 立即数模式
    IMM_PATTERN = re.compile(r"^-?(?:0x[0-9a-fA-F]+|0b[01]+|\d+)$")

    # 标签引用模式（在操作数中）
    LABEL_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

    # 偏移(寄存器) 模式，如 0(sp), -4(s0), symbol(t0)
    OFFSET_REG_PATTERN = re.compile(
        r"^(-?(?:0x[0-9a-fA-F]+|\d+)|[a-zA-Z_][a-zA-Z0-9_]*)\((\w+)\)(!?)$"
    )

    # CSR 名称或数字
    CSR_PATTERN = re.compile(r"^(?:0x[0-9a-fA-F]+|\d+|[a-zA-Z_][a-zA-Z0-9_]*)$")

    def __init__(self, verbose: bool = False):
        """
        初始化解析器

        Args:
            verbose: 是否输出详细日志
        """
        self.verbose = verbose

    def parse_file(self, filepath: Union[str, Path]) -> List[Instruction]:
        """
        解析汇编文件

        Args:
            filepath: 汇编文件路径

        Returns:
            指令列表（包含标签）
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"汇编文件不存在: {filepath}")

        instructions = []
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, 1):
                parsed = self.parse_line(line, line_no)
                if parsed:
                    if isinstance(parsed, list):
                        instructions.extend(parsed)
                    else:
                        instructions.append(parsed)

        if self.verbose:
            print(f"解析完成: {len(instructions)} 条指令/标签")

        return instructions

    def parse_line(
        self, line: str, line_no: int
    ) -> Optional[Union[Instruction, List[Instruction]]]:
        """
        解析单行

        Args:
            line: 行文本
            line_no: 行号

        Returns:
            Instruction 对象，或多个对象（如同一行有标签和指令），或 None
        """
        # 去除注释
        line = self._strip_comment(line)
        line = line.strip()

        if not line:
            return None

        # 跳过预处理器指令和 GCC 行标记（行首 #）
        if line.startswith("#"):
            return None

        results = []

        # 处理同一行的标签和指令，如 "label: instruction"
        if ":" in line:
            parts = line.split(":", 1)
            label_part = parts[0].strip()
            rest = parts[1].strip() if len(parts) > 1 else ""

            # 检查是否是数字标签（局部标签）或普通标签
            if label_part and (
                label_part.isidentifier()
                or label_part.isdigit()
                or label_part[0].isdigit()
            ):
                # 创建标签指令
                label_instr = Instruction(
                    line_no=line_no,
                    raw_text=f"{label_part}:",
                    mnemonic="",
                    operands=[],
                    instr_format=InstrFormat.LABEL,
                    category=InstrCategory.DIRECTIVE,
                    label=label_part,
                )
                results.append(label_instr)

            # 处理剩余部分
            if rest:
                parsed = self._parse_instruction(rest, line_no)
                if parsed:
                    results.append(parsed)
        else:
            # 没有标签，直接解析
            parsed = self._parse_instruction(line, line_no)
            if parsed:
                results.append(parsed)

        if not results:
            return None
        elif len(results) == 1:
            return results[0]
        else:
            return results

    def _strip_comment(self, line: str) -> str:
        """去除行注释"""
        # 处理 # 和 // 风格的注释
        # 行首 # 可能是预处理器指令或 GCC 行标记，不作为注释处理
        for comment_char in ["#", "//"]:
            idx = line.find(comment_char)
            if idx != -1:
                if comment_char == "#" and line.lstrip().startswith("#"):
                    # 行首 #：跳过，保留整行（预处理器指令）
                    continue
                line = line[:idx]
        return line

    def _parse_instruction(self, text: str, line_no: int) -> Optional[Instruction]:
        """解析指令文本"""
        text = text.strip()
        if not text:
            return None

        # 分割助记符和操作数
        parts = text.split(None, 1)
        mnemonic = parts[0].lower()
        operand_str = parts[1].strip() if len(parts) > 1 else ""

        # 检查是否为汇编指令（directive）
        if mnemonic.startswith("."):
            return Instruction(
                line_no=line_no,
                raw_text=text,
                mnemonic=mnemonic,
                operands=self._split_operands(operand_str),
                instr_format=InstrFormat.DIRECTIVE,
                category=InstrCategory.DIRECTIVE,
            )

        # 获取指令分类
        instr_format, category = InstructionClassifier.classify(mnemonic)

        # 解析操作数
        operands = self._split_operands(operand_str)
        parsed_operands = self._parse_operands(mnemonic, operands, instr_format)

        # 判断是否为终结指令
        is_terminator = InstructionClassifier.is_terminator(mnemonic)

        return Instruction(
            line_no=line_no,
            raw_text=text,
            mnemonic=mnemonic,
            operands=operands,
            instr_format=instr_format,
            category=category,
            rd=parsed_operands.get("rd"),
            rs1=parsed_operands.get("rs1"),
            rs2=parsed_operands.get("rs2"),
            rs3=parsed_operands.get("rs3"),
            imm=parsed_operands.get("imm"),
            label=parsed_operands.get("label"),
            is_terminator=is_terminator,
        )

    def _split_operands(self, operand_str: str) -> List[str]:
        """分割操作数（处理逗号分隔）"""
        if not operand_str:
            return []

        # 简单按逗号分割，去除空白
        parts = [p.strip() for p in operand_str.split(",")]
        return [p for p in parts if p]

    def _parse_operands(
        self, mnemonic: str, operands: List[str], instr_format: InstrFormat
    ) -> Dict:
        """
        解析操作数，提取 rd, rs1, rs2, imm, label

        根据指令格式确定操作数含义
        """
        result = {}

        if not operands:
            return result

        # 标准化寄存器名
        def normalize_reg(reg: str) -> Optional[str]:
            reg = reg.lower().strip()
            if reg in REGISTER_ABI_TO_X:
                return REGISTER_ABI_TO_X[reg]
            if reg in FLOAT_REGISTER_ABI_TO_F:
                return FLOAT_REGISTER_ABI_TO_F[reg]
            if re.match(r"^x\d+$", reg) or re.match(r"^f\d+$", reg):
                return reg
            return None

        # 解析立即数
        def parse_imm(imm_str: str) -> Optional[Union[int, str]]:
            imm_str = imm_str.strip()
            try:
                if imm_str.startswith("0x") or imm_str.startswith("-0x"):
                    return int(imm_str, 16)
                elif imm_str.startswith("0b"):
                    return int(imm_str, 2)
                else:
                    return int(imm_str)
            except ValueError:
                # 可能是符号（标签）
                return imm_str

        # 根据指令格式解析
        if instr_format == InstrFormat.R_TYPE:
            # rd, rs1, rs2
            if len(operands) >= 3:
                result["rd"] = normalize_reg(operands[0])
                result["rs1"] = normalize_reg(operands[1])
                result["rs2"] = normalize_reg(operands[2])
            elif len(operands) == 2:
                # 某些伪指令如 neg rd, rs
                result["rd"] = normalize_reg(operands[0])
                result["rs1"] = normalize_reg(operands[1])

        elif instr_format == InstrFormat.I_TYPE:
            # 多种格式:
            # - rd, rs1, imm (addi, ori, etc.)
            # - rd, imm(rs1) (load)
            # - rd, rs1, rs2 (某些伪指令)
            if len(operands) >= 2:
                result["rd"] = normalize_reg(operands[0])

                # 检查是否是 offset(reg) 格式
                match = self.OFFSET_REG_PATTERN.match(operands[1])
                if match:
                    result["imm"] = parse_imm(match.group(1))
                    result["rs1"] = normalize_reg(match.group(2))
                elif len(operands) >= 3:
                    result["rs1"] = normalize_reg(operands[1])
                    result["imm"] = parse_imm(operands[2])
                else:
                    # 可能是 jalr rd, rs1 格式
                    result["rs1"] = normalize_reg(operands[1])

        elif instr_format == InstrFormat.S_TYPE:
            # rs2, offset(rs1) - store
            if len(operands) >= 2:
                result["rs2"] = normalize_reg(operands[0])
                match = self.OFFSET_REG_PATTERN.match(operands[1])
                if match:
                    result["imm"] = parse_imm(match.group(1))
                    result["rs1"] = normalize_reg(match.group(2))

        elif instr_format == InstrFormat.B_TYPE:
            # rs1, rs2, label 或 rs1, label (beqz/bnez)
            if len(operands) >= 3:
                result["rs1"] = normalize_reg(operands[0])
                result["rs2"] = normalize_reg(operands[1])
                result["label"] = operands[2]
            elif len(operands) >= 2:
                result["rs1"] = normalize_reg(operands[0])
                # 第二个可能是立即数或标签
                second = operands[1]
                if normalize_reg(second):
                    result["rs2"] = normalize_reg(second)
                else:
                    result["label"] = second

        elif instr_format == InstrFormat.U_TYPE:
            # rd, imm
            if len(operands) >= 2:
                result["rd"] = normalize_reg(operands[0])
                result["imm"] = parse_imm(operands[1])

        elif instr_format == InstrFormat.J_TYPE:
            # rd, label 或 label (j 伪指令)
            if len(operands) >= 2:
                reg = normalize_reg(operands[0])
                if reg:
                    result["rd"] = reg
                    result["label"] = operands[1]
                else:
                    result["label"] = operands[0]
            elif len(operands) == 1:
                result["label"] = operands[0]

        elif instr_format == InstrFormat.CSR_TYPE:
            # csrrw rd, csr, rs1
            # csrrwi rd, csr, imm
            if len(operands) >= 3:
                result["rd"] = normalize_reg(operands[0])
                result["imm"] = parse_imm(operands[1])  # CSR 地址
                reg = normalize_reg(operands[2])
                if reg:
                    result["rs1"] = reg
            elif len(operands) >= 2:
                # csrr rd, csr 或 csrw csr, rs1
                first_reg = normalize_reg(operands[0])
                if first_reg:
                    result["rd"] = first_reg
                    result["imm"] = parse_imm(operands[1])
                else:
                    result["imm"] = parse_imm(operands[0])
                    result["rs1"] = normalize_reg(operands[1])

        # RV32C 压缩格式
        elif instr_format in (
            InstrFormat.CR_TYPE,
            InstrFormat.CI_TYPE,
            InstrFormat.CSS_TYPE,
            InstrFormat.CIW_TYPE,
            InstrFormat.CL_TYPE,
            InstrFormat.CS_TYPE,
            InstrFormat.CB_TYPE,
            InstrFormat.CJ_TYPE,
        ):
            self._parse_compressed_operands(
                mnemonic, operands, result, normalize_reg, parse_imm
            )

        # Xpulp 扩展
        elif instr_format == InstrFormat.XPULP_TYPE:
            self._parse_xpulp_operands(
                mnemonic, operands, result, normalize_reg, parse_imm
            )

        # 伪指令
        elif instr_format == InstrFormat.PSEUDO:
            self._parse_pseudo_operands(
                mnemonic, operands, result, normalize_reg, parse_imm
            )

        return result

    def _parse_compressed_operands(
        self, mnemonic: str, operands: List[str], result: Dict, normalize_reg, parse_imm
    ):
        """解析压缩指令操作数"""
        mnemonic_lower = mnemonic.lower()

        if mnemonic_lower in ("c.add", "c.mv", "c.sub", "c.and", "c.or", "c.xor"):
            # rd, rs2
            if len(operands) >= 2:
                result["rd"] = normalize_reg(operands[0])
                result["rs2"] = normalize_reg(operands[1])

        elif mnemonic_lower in (
            "c.addi",
            "c.slli",
            "c.srli",
            "c.srai",
            "c.andi",
            "c.li",
        ):
            # rd, imm
            if len(operands) >= 2:
                result["rd"] = normalize_reg(operands[0])
                result["imm"] = parse_imm(operands[1])

        elif mnemonic_lower in ("c.lw", "c.lwsp", "c.flw", "c.flwsp"):
            # rd, offset(rs1)
            if len(operands) >= 2:
                result["rd"] = normalize_reg(operands[0])
                match = self.OFFSET_REG_PATTERN.match(operands[1])
                if match:
                    result["imm"] = parse_imm(match.group(1))
                    result["rs1"] = normalize_reg(match.group(2))

        elif mnemonic_lower in ("c.sw", "c.swsp", "c.fsw", "c.fswsp"):
            # rs2, offset(rs1)
            if len(operands) >= 2:
                result["rs2"] = normalize_reg(operands[0])
                match = self.OFFSET_REG_PATTERN.match(operands[1])
                if match:
                    result["imm"] = parse_imm(match.group(1))
                    result["rs1"] = normalize_reg(match.group(2))

        elif mnemonic_lower in ("c.beqz", "c.bnez"):
            # rs1, label
            if len(operands) >= 2:
                result["rs1"] = normalize_reg(operands[0])
                result["label"] = operands[1]

        elif mnemonic_lower in ("c.j", "c.jal"):
            # label
            if len(operands) >= 1:
                result["label"] = operands[0]

        elif mnemonic_lower in ("c.jr", "c.jalr"):
            # rs1
            if len(operands) >= 1:
                result["rs1"] = normalize_reg(operands[0])

        elif mnemonic_lower == "c.lui":
            # rd, imm
            if len(operands) >= 2:
                result["rd"] = normalize_reg(operands[0])
                result["imm"] = parse_imm(operands[1])

    def _parse_xpulp_operands(
        self, mnemonic: str, operands: List[str], result: Dict, normalize_reg, parse_imm
    ):
        """解析 Xpulp 扩展指令操作数"""
        mnemonic_lower = mnemonic.lower()

        # 后递增加载: cv.lw rd, imm(rs1!) 或 cv.lw rd, rs2(rs1!)
        if mnemonic_lower.startswith("cv.l"):
            if len(operands) >= 2:
                result["rd"] = normalize_reg(operands[0])
                match = self.OFFSET_REG_PATTERN.match(operands[1])
                if match:
                    offset = match.group(1)
                    base_reg = match.group(2)
                    post_inc = match.group(3) == "!"
                    result["rs1"] = normalize_reg(base_reg)
                    # 检查 offset 是寄存器还是立即数
                    if normalize_reg(offset):
                        result["rs2"] = normalize_reg(offset)
                    else:
                        result["imm"] = parse_imm(offset)

        # 后递增存储
        elif mnemonic_lower.startswith("cv.s"):
            if len(operands) >= 2:
                result["rs2"] = normalize_reg(operands[0])
                match = self.OFFSET_REG_PATTERN.match(operands[1])
                if match:
                    offset = match.group(1)
                    base_reg = match.group(2)
                    result["rs1"] = normalize_reg(base_reg)
                    if normalize_reg(offset):
                        result["rs3"] = normalize_reg(offset)
                    else:
                        result["imm"] = parse_imm(offset)

        # MAC 指令: cv.mac rd, rs1, rs2
        elif mnemonic_lower in ("cv.mac", "cv.msu"):
            if len(operands) >= 3:
                result["rd"] = normalize_reg(operands[0])
                result["rs1"] = normalize_reg(operands[1])
                result["rs2"] = normalize_reg(operands[2])

        # 位操作: cv.extract rd, rs1, imm1, imm2
        elif mnemonic_lower.startswith("cv.extract") or mnemonic_lower.startswith(
            "cv.insert"
        ):
            if len(operands) >= 2:
                result["rd"] = normalize_reg(operands[0])
                result["rs1"] = normalize_reg(operands[1])
                if len(operands) >= 4:
                    result["imm"] = parse_imm(operands[2])

        # 分支扩展
        elif mnemonic_lower in ("cv.beqimm", "cv.bneimm"):
            if len(operands) >= 3:
                result["rs1"] = normalize_reg(operands[0])
                result["imm"] = parse_imm(operands[1])
                result["label"] = operands[2]

        # 默认 R-type 格式
        else:
            if len(operands) >= 3:
                result["rd"] = normalize_reg(operands[0])
                result["rs1"] = normalize_reg(operands[1])
                result["rs2"] = normalize_reg(operands[2])
            elif len(operands) >= 2:
                result["rd"] = normalize_reg(operands[0])
                result["rs1"] = normalize_reg(operands[1])

    def _parse_pseudo_operands(
        self, mnemonic: str, operands: List[str], result: Dict, normalize_reg, parse_imm
    ):
        """解析伪指令操作数"""
        mnemonic_lower = mnemonic.lower()

        if mnemonic_lower in ("li", "la"):
            # rd, imm/symbol
            if len(operands) >= 2:
                result["rd"] = normalize_reg(operands[0])
                imm = parse_imm(operands[1])
                if isinstance(imm, str):
                    result["label"] = imm
                else:
                    result["imm"] = imm

        elif mnemonic_lower == "mv":
            # rd, rs1
            if len(operands) >= 2:
                result["rd"] = normalize_reg(operands[0])
                result["rs1"] = normalize_reg(operands[1])

        elif mnemonic_lower in ("not", "neg"):
            # rd, rs1
            if len(operands) >= 2:
                result["rd"] = normalize_reg(operands[0])
                result["rs1"] = normalize_reg(operands[1])

        elif mnemonic_lower in ("call", "tail"):
            # symbol
            if len(operands) >= 1:
                result["label"] = operands[0]

        elif mnemonic_lower in ("seqz", "snez", "sltz", "sgtz"):
            # rd, rs1
            if len(operands) >= 2:
                result["rd"] = normalize_reg(operands[0])
                result["rs1"] = normalize_reg(operands[1])
