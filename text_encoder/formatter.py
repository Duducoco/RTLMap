"""ASM 指令文本格式化"""


class InstructionFormatter:
    """将 ASM JSON 节点的 instructions 列表格式化为文本字符串"""

    @staticmethod
    def format_instruction(instr: dict) -> str:
        """单条指令 → "addi x1, x2, 5" """
        mnemonic = instr.get("mnemonic", "")
        operands = instr.get("operands", [])
        return f"{mnemonic} {', '.join(operands)}" if operands else mnemonic

    @staticmethod
    def format_block(instructions: list[dict]) -> str:
        """基本块所有指令 → 换行拼接的文本"""
        if not instructions:
            return "nop"  # 空块用 nop 占位
        return "\n".join(
            InstructionFormatter.format_instruction(i) for i in instructions
        )
