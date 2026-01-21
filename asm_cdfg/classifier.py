#!/usr/bin/env python3
"""
RISC-V 指令分类器

可扩展的指令分类器，支持 RV32IMFC + Xpulp 扩展。
使用注册表模式便于后续添加新扩展。
"""

from typing import Dict, Tuple, Optional
from .asm_types import InstrFormat, InstrCategory


class InstructionClassifier:
    """可扩展的 RISC-V 指令分类器"""

    # 动态注册的指令扩展
    _registry: Dict[str, Tuple[InstrFormat, InstrCategory]] = {}

    # ========== RV32I 基本指令集 ==========
    RV32I_INSTRS: Dict[str, Tuple[InstrFormat, InstrCategory]] = {
        # R-type 算术逻辑
        "add": (InstrFormat.R_TYPE, InstrCategory.ARITHMETIC),
        "sub": (InstrFormat.R_TYPE, InstrCategory.ARITHMETIC),
        "and": (InstrFormat.R_TYPE, InstrCategory.LOGIC),
        "or": (InstrFormat.R_TYPE, InstrCategory.LOGIC),
        "xor": (InstrFormat.R_TYPE, InstrCategory.LOGIC),
        "sll": (InstrFormat.R_TYPE, InstrCategory.SHIFT),
        "srl": (InstrFormat.R_TYPE, InstrCategory.SHIFT),
        "sra": (InstrFormat.R_TYPE, InstrCategory.SHIFT),
        "slt": (InstrFormat.R_TYPE, InstrCategory.COMPARE),
        "sltu": (InstrFormat.R_TYPE, InstrCategory.COMPARE),
        # I-type 算术逻辑
        "addi": (InstrFormat.I_TYPE, InstrCategory.ARITHMETIC),
        "andi": (InstrFormat.I_TYPE, InstrCategory.LOGIC),
        "ori": (InstrFormat.I_TYPE, InstrCategory.LOGIC),
        "xori": (InstrFormat.I_TYPE, InstrCategory.LOGIC),
        "slti": (InstrFormat.I_TYPE, InstrCategory.COMPARE),
        "sltiu": (InstrFormat.I_TYPE, InstrCategory.COMPARE),
        "slli": (InstrFormat.I_TYPE, InstrCategory.SHIFT),
        "srli": (InstrFormat.I_TYPE, InstrCategory.SHIFT),
        "srai": (InstrFormat.I_TYPE, InstrCategory.SHIFT),
        # Load
        "lb": (InstrFormat.I_TYPE, InstrCategory.LOAD),
        "lh": (InstrFormat.I_TYPE, InstrCategory.LOAD),
        "lw": (InstrFormat.I_TYPE, InstrCategory.LOAD),
        "lbu": (InstrFormat.I_TYPE, InstrCategory.LOAD),
        "lhu": (InstrFormat.I_TYPE, InstrCategory.LOAD),
        # Store
        "sb": (InstrFormat.S_TYPE, InstrCategory.STORE),
        "sh": (InstrFormat.S_TYPE, InstrCategory.STORE),
        "sw": (InstrFormat.S_TYPE, InstrCategory.STORE),
        # Branch
        "beq": (InstrFormat.B_TYPE, InstrCategory.BRANCH),
        "bne": (InstrFormat.B_TYPE, InstrCategory.BRANCH),
        "blt": (InstrFormat.B_TYPE, InstrCategory.BRANCH),
        "bge": (InstrFormat.B_TYPE, InstrCategory.BRANCH),
        "bltu": (InstrFormat.B_TYPE, InstrCategory.BRANCH),
        "bgeu": (InstrFormat.B_TYPE, InstrCategory.BRANCH),
        # 分支伪指令
        "beqz": (InstrFormat.B_TYPE, InstrCategory.BRANCH),
        "bnez": (InstrFormat.B_TYPE, InstrCategory.BRANCH),
        "blez": (InstrFormat.B_TYPE, InstrCategory.BRANCH),
        "bgez": (InstrFormat.B_TYPE, InstrCategory.BRANCH),
        "bltz": (InstrFormat.B_TYPE, InstrCategory.BRANCH),
        "bgtz": (InstrFormat.B_TYPE, InstrCategory.BRANCH),
        # Jump
        "jal": (InstrFormat.J_TYPE, InstrCategory.JUMP),
        "jalr": (InstrFormat.I_TYPE, InstrCategory.JUMP),
        "j": (InstrFormat.J_TYPE, InstrCategory.JUMP),  # 伪指令
        "jr": (InstrFormat.I_TYPE, InstrCategory.JUMP),  # 伪指令
        "ret": (InstrFormat.I_TYPE, InstrCategory.JUMP),  # 伪指令
        # U-type
        "lui": (InstrFormat.U_TYPE, InstrCategory.ARITHMETIC),
        "auipc": (InstrFormat.U_TYPE, InstrCategory.ARITHMETIC),
        # System
        "ecall": (InstrFormat.I_TYPE, InstrCategory.SYSTEM),
        "ebreak": (InstrFormat.I_TYPE, InstrCategory.SYSTEM),
        "fence": (InstrFormat.I_TYPE, InstrCategory.SYSTEM),
        "fence.i": (InstrFormat.I_TYPE, InstrCategory.SYSTEM),
        "mret": (InstrFormat.I_TYPE, InstrCategory.SYSTEM),
        "sret": (InstrFormat.I_TYPE, InstrCategory.SYSTEM),
        "wfi": (InstrFormat.I_TYPE, InstrCategory.SYSTEM),
        "dret": (InstrFormat.I_TYPE, InstrCategory.SYSTEM),
        "nop": (InstrFormat.I_TYPE, InstrCategory.NOP),
        # CSR
        "csrrw": (InstrFormat.CSR_TYPE, InstrCategory.CSR),
        "csrrs": (InstrFormat.CSR_TYPE, InstrCategory.CSR),
        "csrrc": (InstrFormat.CSR_TYPE, InstrCategory.CSR),
        "csrrwi": (InstrFormat.CSR_TYPE, InstrCategory.CSR),
        "csrrsi": (InstrFormat.CSR_TYPE, InstrCategory.CSR),
        "csrrci": (InstrFormat.CSR_TYPE, InstrCategory.CSR),
        # CSR 伪指令
        "csrr": (InstrFormat.CSR_TYPE, InstrCategory.CSR),
        "csrw": (InstrFormat.CSR_TYPE, InstrCategory.CSR),
        "csrs": (InstrFormat.CSR_TYPE, InstrCategory.CSR),
        "csrc": (InstrFormat.CSR_TYPE, InstrCategory.CSR),
        "csrwi": (InstrFormat.CSR_TYPE, InstrCategory.CSR),
        "csrsi": (InstrFormat.CSR_TYPE, InstrCategory.CSR),
        "csrci": (InstrFormat.CSR_TYPE, InstrCategory.CSR),
    }

    # ========== RV32M 乘除扩展 ==========
    RV32M_INSTRS: Dict[str, Tuple[InstrFormat, InstrCategory]] = {
        "mul": (InstrFormat.R_TYPE, InstrCategory.ARITHMETIC),
        "mulh": (InstrFormat.R_TYPE, InstrCategory.ARITHMETIC),
        "mulhsu": (InstrFormat.R_TYPE, InstrCategory.ARITHMETIC),
        "mulhu": (InstrFormat.R_TYPE, InstrCategory.ARITHMETIC),
        "div": (InstrFormat.R_TYPE, InstrCategory.ARITHMETIC),
        "divu": (InstrFormat.R_TYPE, InstrCategory.ARITHMETIC),
        "rem": (InstrFormat.R_TYPE, InstrCategory.ARITHMETIC),
        "remu": (InstrFormat.R_TYPE, InstrCategory.ARITHMETIC),
    }

    # ========== RV32F 单精度浮点扩展 ==========
    RV32F_INSTRS: Dict[str, Tuple[InstrFormat, InstrCategory]] = {
        # 浮点加载/存储
        "flw": (InstrFormat.I_TYPE, InstrCategory.LOAD),
        "fsw": (InstrFormat.S_TYPE, InstrCategory.STORE),
        # 浮点算术
        "fadd.s": (InstrFormat.R_TYPE, InstrCategory.FLOAT),
        "fsub.s": (InstrFormat.R_TYPE, InstrCategory.FLOAT),
        "fmul.s": (InstrFormat.R_TYPE, InstrCategory.FLOAT),
        "fdiv.s": (InstrFormat.R_TYPE, InstrCategory.FLOAT),
        "fsqrt.s": (InstrFormat.R_TYPE, InstrCategory.FLOAT),
        # 浮点乘加
        "fmadd.s": (InstrFormat.R_TYPE, InstrCategory.FLOAT),
        "fmsub.s": (InstrFormat.R_TYPE, InstrCategory.FLOAT),
        "fnmsub.s": (InstrFormat.R_TYPE, InstrCategory.FLOAT),
        "fnmadd.s": (InstrFormat.R_TYPE, InstrCategory.FLOAT),
        # 浮点比较
        "feq.s": (InstrFormat.R_TYPE, InstrCategory.COMPARE),
        "flt.s": (InstrFormat.R_TYPE, InstrCategory.COMPARE),
        "fle.s": (InstrFormat.R_TYPE, InstrCategory.COMPARE),
        "fmin.s": (InstrFormat.R_TYPE, InstrCategory.FLOAT),
        "fmax.s": (InstrFormat.R_TYPE, InstrCategory.FLOAT),
        # 浮点转换
        "fcvt.w.s": (InstrFormat.R_TYPE, InstrCategory.FLOAT),
        "fcvt.wu.s": (InstrFormat.R_TYPE, InstrCategory.FLOAT),
        "fcvt.s.w": (InstrFormat.R_TYPE, InstrCategory.FLOAT),
        "fcvt.s.wu": (InstrFormat.R_TYPE, InstrCategory.FLOAT),
        # 浮点移动
        "fmv.x.w": (InstrFormat.R_TYPE, InstrCategory.FLOAT),
        "fmv.w.x": (InstrFormat.R_TYPE, InstrCategory.FLOAT),
        # 浮点符号操作
        "fsgnj.s": (InstrFormat.R_TYPE, InstrCategory.FLOAT),
        "fsgnjn.s": (InstrFormat.R_TYPE, InstrCategory.FLOAT),
        "fsgnjx.s": (InstrFormat.R_TYPE, InstrCategory.FLOAT),
        # 浮点分类
        "fclass.s": (InstrFormat.R_TYPE, InstrCategory.FLOAT),
    }

    # ========== RV32C 压缩扩展 ==========
    RV32C_INSTRS: Dict[str, Tuple[InstrFormat, InstrCategory]] = {
        # 算术
        "c.add": (InstrFormat.CR_TYPE, InstrCategory.ARITHMETIC),
        "c.addi": (InstrFormat.CI_TYPE, InstrCategory.ARITHMETIC),
        "c.addi16sp": (InstrFormat.CI_TYPE, InstrCategory.ARITHMETIC),
        "c.addi4spn": (InstrFormat.CIW_TYPE, InstrCategory.ARITHMETIC),
        "c.sub": (InstrFormat.CR_TYPE, InstrCategory.ARITHMETIC),
        # 逻辑
        "c.and": (InstrFormat.CR_TYPE, InstrCategory.LOGIC),
        "c.or": (InstrFormat.CR_TYPE, InstrCategory.LOGIC),
        "c.xor": (InstrFormat.CR_TYPE, InstrCategory.LOGIC),
        "c.andi": (InstrFormat.CI_TYPE, InstrCategory.LOGIC),
        # 移位
        "c.slli": (InstrFormat.CI_TYPE, InstrCategory.SHIFT),
        "c.srli": (InstrFormat.CI_TYPE, InstrCategory.SHIFT),
        "c.srai": (InstrFormat.CI_TYPE, InstrCategory.SHIFT),
        # 加载
        "c.lw": (InstrFormat.CL_TYPE, InstrCategory.LOAD),
        "c.lwsp": (InstrFormat.CI_TYPE, InstrCategory.LOAD),
        "c.flw": (InstrFormat.CL_TYPE, InstrCategory.LOAD),
        "c.flwsp": (InstrFormat.CI_TYPE, InstrCategory.LOAD),
        # 存储
        "c.sw": (InstrFormat.CS_TYPE, InstrCategory.STORE),
        "c.swsp": (InstrFormat.CSS_TYPE, InstrCategory.STORE),
        "c.fsw": (InstrFormat.CS_TYPE, InstrCategory.STORE),
        "c.fswsp": (InstrFormat.CSS_TYPE, InstrCategory.STORE),
        # 分支
        "c.beqz": (InstrFormat.CB_TYPE, InstrCategory.BRANCH),
        "c.bnez": (InstrFormat.CB_TYPE, InstrCategory.BRANCH),
        # 跳转
        "c.j": (InstrFormat.CJ_TYPE, InstrCategory.JUMP),
        "c.jal": (InstrFormat.CJ_TYPE, InstrCategory.JUMP),
        "c.jr": (InstrFormat.CR_TYPE, InstrCategory.JUMP),
        "c.jalr": (InstrFormat.CR_TYPE, InstrCategory.JUMP),
        # 其他
        "c.li": (InstrFormat.CI_TYPE, InstrCategory.ARITHMETIC),
        "c.lui": (InstrFormat.CI_TYPE, InstrCategory.ARITHMETIC),
        "c.mv": (InstrFormat.CR_TYPE, InstrCategory.ARITHMETIC),
        "c.nop": (InstrFormat.CI_TYPE, InstrCategory.NOP),
        "c.ebreak": (InstrFormat.CR_TYPE, InstrCategory.SYSTEM),
    }

    # ========== Xpulp 扩展（CV32E40P 特有） ==========
    XPULP_INSTRS: Dict[str, Tuple[InstrFormat, InstrCategory]] = {
        # 硬件循环
        "cv.starti": (InstrFormat.XPULP_TYPE, InstrCategory.HWLOOP),
        "cv.endi": (InstrFormat.XPULP_TYPE, InstrCategory.HWLOOP),
        "cv.counti": (InstrFormat.XPULP_TYPE, InstrCategory.HWLOOP),
        "cv.count": (InstrFormat.XPULP_TYPE, InstrCategory.HWLOOP),
        "cv.setupi": (InstrFormat.XPULP_TYPE, InstrCategory.HWLOOP),
        "cv.setup": (InstrFormat.XPULP_TYPE, InstrCategory.HWLOOP),
        # MAC (乘累加)
        "cv.mac": (InstrFormat.XPULP_TYPE, InstrCategory.MAC),
        "cv.msu": (InstrFormat.XPULP_TYPE, InstrCategory.MAC),
        # 16-bit 乘法
        "cv.muluN": (InstrFormat.XPULP_TYPE, InstrCategory.MAC),
        "cv.mulhhuN": (InstrFormat.XPULP_TYPE, InstrCategory.MAC),
        "cv.mulsN": (InstrFormat.XPULP_TYPE, InstrCategory.MAC),
        "cv.mulhhsN": (InstrFormat.XPULP_TYPE, InstrCategory.MAC),
        "cv.muluRN": (InstrFormat.XPULP_TYPE, InstrCategory.MAC),
        "cv.mulhhuRN": (InstrFormat.XPULP_TYPE, InstrCategory.MAC),
        "cv.mulsRN": (InstrFormat.XPULP_TYPE, InstrCategory.MAC),
        "cv.mulhhsRN": (InstrFormat.XPULP_TYPE, InstrCategory.MAC),
        # 16-bit MAC
        "cv.macuN": (InstrFormat.XPULP_TYPE, InstrCategory.MAC),
        "cv.machhuN": (InstrFormat.XPULP_TYPE, InstrCategory.MAC),
        "cv.macsN": (InstrFormat.XPULP_TYPE, InstrCategory.MAC),
        "cv.machhsN": (InstrFormat.XPULP_TYPE, InstrCategory.MAC),
        "cv.macuRN": (InstrFormat.XPULP_TYPE, InstrCategory.MAC),
        "cv.machhuRN": (InstrFormat.XPULP_TYPE, InstrCategory.MAC),
        "cv.macsRN": (InstrFormat.XPULP_TYPE, InstrCategory.MAC),
        "cv.machhsRN": (InstrFormat.XPULP_TYPE, InstrCategory.MAC),
        # 位操作
        "cv.extract": (InstrFormat.XPULP_TYPE, InstrCategory.BITMANIP),
        "cv.extractu": (InstrFormat.XPULP_TYPE, InstrCategory.BITMANIP),
        "cv.extractr": (InstrFormat.XPULP_TYPE, InstrCategory.BITMANIP),
        "cv.extractur": (InstrFormat.XPULP_TYPE, InstrCategory.BITMANIP),
        "cv.insert": (InstrFormat.XPULP_TYPE, InstrCategory.BITMANIP),
        "cv.insertr": (InstrFormat.XPULP_TYPE, InstrCategory.BITMANIP),
        "cv.bclr": (InstrFormat.XPULP_TYPE, InstrCategory.BITMANIP),
        "cv.bclrr": (InstrFormat.XPULP_TYPE, InstrCategory.BITMANIP),
        "cv.bset": (InstrFormat.XPULP_TYPE, InstrCategory.BITMANIP),
        "cv.bsetr": (InstrFormat.XPULP_TYPE, InstrCategory.BITMANIP),
        "cv.ff1": (InstrFormat.XPULP_TYPE, InstrCategory.BITMANIP),
        "cv.fl1": (InstrFormat.XPULP_TYPE, InstrCategory.BITMANIP),
        "cv.clb": (InstrFormat.XPULP_TYPE, InstrCategory.BITMANIP),
        "cv.cnt": (InstrFormat.XPULP_TYPE, InstrCategory.BITMANIP),
        "cv.ror": (InstrFormat.XPULP_TYPE, InstrCategory.BITMANIP),
        "cv.bitrev": (InstrFormat.XPULP_TYPE, InstrCategory.BITMANIP),
        # ALU 扩展
        "cv.abs": (InstrFormat.XPULP_TYPE, InstrCategory.ARITHMETIC),
        "cv.slet": (InstrFormat.XPULP_TYPE, InstrCategory.COMPARE),
        "cv.sletu": (InstrFormat.XPULP_TYPE, InstrCategory.COMPARE),
        "cv.min": (InstrFormat.XPULP_TYPE, InstrCategory.COMPARE),
        "cv.minu": (InstrFormat.XPULP_TYPE, InstrCategory.COMPARE),
        "cv.max": (InstrFormat.XPULP_TYPE, InstrCategory.COMPARE),
        "cv.maxu": (InstrFormat.XPULP_TYPE, InstrCategory.COMPARE),
        "cv.exths": (InstrFormat.XPULP_TYPE, InstrCategory.ARITHMETIC),
        "cv.exthz": (InstrFormat.XPULP_TYPE, InstrCategory.ARITHMETIC),
        "cv.extbs": (InstrFormat.XPULP_TYPE, InstrCategory.ARITHMETIC),
        "cv.extbz": (InstrFormat.XPULP_TYPE, InstrCategory.ARITHMETIC),
        "cv.clip": (InstrFormat.XPULP_TYPE, InstrCategory.ARITHMETIC),
        "cv.clipu": (InstrFormat.XPULP_TYPE, InstrCategory.ARITHMETIC),
        "cv.clipr": (InstrFormat.XPULP_TYPE, InstrCategory.ARITHMETIC),
        "cv.clipur": (InstrFormat.XPULP_TYPE, InstrCategory.ARITHMETIC),
        # Add/Sub 带归一化
        "cv.addN": (InstrFormat.XPULP_TYPE, InstrCategory.ARITHMETIC),
        "cv.adduN": (InstrFormat.XPULP_TYPE, InstrCategory.ARITHMETIC),
        "cv.addRN": (InstrFormat.XPULP_TYPE, InstrCategory.ARITHMETIC),
        "cv.adduRN": (InstrFormat.XPULP_TYPE, InstrCategory.ARITHMETIC),
        "cv.subN": (InstrFormat.XPULP_TYPE, InstrCategory.ARITHMETIC),
        "cv.subuN": (InstrFormat.XPULP_TYPE, InstrCategory.ARITHMETIC),
        "cv.subRN": (InstrFormat.XPULP_TYPE, InstrCategory.ARITHMETIC),
        "cv.subuRN": (InstrFormat.XPULP_TYPE, InstrCategory.ARITHMETIC),
        "cv.addNr": (InstrFormat.XPULP_TYPE, InstrCategory.ARITHMETIC),
        "cv.adduNr": (InstrFormat.XPULP_TYPE, InstrCategory.ARITHMETIC),
        "cv.addRNr": (InstrFormat.XPULP_TYPE, InstrCategory.ARITHMETIC),
        "cv.adduRNr": (InstrFormat.XPULP_TYPE, InstrCategory.ARITHMETIC),
        "cv.subNr": (InstrFormat.XPULP_TYPE, InstrCategory.ARITHMETIC),
        "cv.subuNr": (InstrFormat.XPULP_TYPE, InstrCategory.ARITHMETIC),
        "cv.subRNr": (InstrFormat.XPULP_TYPE, InstrCategory.ARITHMETIC),
        "cv.subuRNr": (InstrFormat.XPULP_TYPE, InstrCategory.ARITHMETIC),
        # 分支扩展
        "cv.beqimm": (InstrFormat.XPULP_TYPE, InstrCategory.BRANCH),
        "cv.bneimm": (InstrFormat.XPULP_TYPE, InstrCategory.BRANCH),
        # 后递增加载
        "cv.lb": (InstrFormat.XPULP_TYPE, InstrCategory.LOAD),
        "cv.lbu": (InstrFormat.XPULP_TYPE, InstrCategory.LOAD),
        "cv.lh": (InstrFormat.XPULP_TYPE, InstrCategory.LOAD),
        "cv.lhu": (InstrFormat.XPULP_TYPE, InstrCategory.LOAD),
        "cv.lw": (InstrFormat.XPULP_TYPE, InstrCategory.LOAD),
        # 后递增存储
        "cv.sb": (InstrFormat.XPULP_TYPE, InstrCategory.STORE),
        "cv.sh": (InstrFormat.XPULP_TYPE, InstrCategory.STORE),
        "cv.sw": (InstrFormat.XPULP_TYPE, InstrCategory.STORE),
        # Event Load
        "cv.elw": (InstrFormat.XPULP_TYPE, InstrCategory.LOAD),
    }

    # ========== Xpulp SIMD 指令（动态生成） ==========
    # SIMD 指令有大量变体，在初始化时生成

    # ========== 伪指令 ==========
    PSEUDO_INSTRS: Dict[str, Tuple[InstrFormat, InstrCategory]] = {
        "li": (InstrFormat.PSEUDO, InstrCategory.ARITHMETIC),
        "la": (InstrFormat.PSEUDO, InstrCategory.ARITHMETIC),
        "mv": (InstrFormat.PSEUDO, InstrCategory.ARITHMETIC),
        "not": (InstrFormat.PSEUDO, InstrCategory.LOGIC),
        "neg": (InstrFormat.PSEUDO, InstrCategory.ARITHMETIC),
        "negw": (InstrFormat.PSEUDO, InstrCategory.ARITHMETIC),
        "sext.w": (InstrFormat.PSEUDO, InstrCategory.ARITHMETIC),
        "seqz": (InstrFormat.PSEUDO, InstrCategory.COMPARE),
        "snez": (InstrFormat.PSEUDO, InstrCategory.COMPARE),
        "sltz": (InstrFormat.PSEUDO, InstrCategory.COMPARE),
        "sgtz": (InstrFormat.PSEUDO, InstrCategory.COMPARE),
        "call": (InstrFormat.PSEUDO, InstrCategory.JUMP),
        "tail": (InstrFormat.PSEUDO, InstrCategory.JUMP),
        # 浮点伪指令
        "fmv.s": (InstrFormat.PSEUDO, InstrCategory.FLOAT),
        "fabs.s": (InstrFormat.PSEUDO, InstrCategory.FLOAT),
        "fneg.s": (InstrFormat.PSEUDO, InstrCategory.FLOAT),
    }

    # ========== 汇编指令（directive） ==========
    DIRECTIVES = {
        ".text",
        ".data",
        ".bss",
        ".section",
        ".globl",
        ".global",
        ".local",
        ".align",
        ".balign",
        ".p2align",
        ".byte",
        ".half",
        ".word",
        ".dword",
        ".4byte",
        ".8byte",
        ".string",
        ".asciz",
        ".ascii",
        ".zero",
        ".space",
        ".fill",
        ".skip",
        ".type",
        ".size",
        ".file",
        ".ident",
        ".option",
        ".attribute",
        ".equ",
        ".set",
        ".equiv",
        ".comm",
        ".lcomm",
        ".include",
        ".incbin",
        ".rept",
        ".endr",
        ".macro",
        ".endm",
        ".if",
        ".else",
        ".endif",
        ".ifdef",
        ".ifndef",
        ".weak",
        ".hidden",
        ".protected",
        ".internal",
        ".pushsection",
        ".popsection",
    }

    @classmethod
    def _init_simd_instructions(cls) -> Dict[str, Tuple[InstrFormat, InstrCategory]]:
        """动态生成 SIMD 指令变体"""
        simd_instrs = {}

        # SIMD 基础操作
        simd_ops = [
            "add",
            "sub",
            "avg",
            "avgu",
            "min",
            "minu",
            "max",
            "maxu",
            "srl",
            "sra",
            "sll",
            "or",
            "xor",
            "and",
            "abs",
        ]

        # SIMD 点积操作
        dot_ops = ["dotup", "dotusp", "dotsp", "sdotup", "sdotusp", "sdotsp"]

        # SIMD 比较操作
        cmp_ops = [
            "cmpeq",
            "cmpne",
            "cmpgt",
            "cmpge",
            "cmplt",
            "cmple",
            "cmpgtu",
            "cmpgeu",
            "cmpltu",
            "cmpleu",
        ]

        # 大小变体
        sizes = [".h", ".b"]
        modes = ["", ".sc", ".sci"]

        # 生成所有变体
        for op in simd_ops:
            for size in sizes:
                for mode in modes:
                    name = f"cv.{op}{mode}{size}"
                    simd_instrs[name] = (InstrFormat.XPULP_TYPE, InstrCategory.SIMD)

        for op in dot_ops:
            for size in sizes:
                for mode in modes:
                    name = f"cv.{op}{mode}{size}"
                    simd_instrs[name] = (InstrFormat.XPULP_TYPE, InstrCategory.SIMD)

        for op in cmp_ops:
            for size in sizes:
                for mode in modes:
                    name = f"cv.{op}{mode}{size}"
                    simd_instrs[name] = (InstrFormat.XPULP_TYPE, InstrCategory.SIMD)

        # 特殊 SIMD 操作
        special_simd = [
            "cv.extract.h",
            "cv.extract.b",
            "cv.extractu.h",
            "cv.extractu.b",
            "cv.insert.h",
            "cv.insert.b",
            "cv.shuffle.h",
            "cv.shuffle.sci.h",
            "cv.shuffle.b",
            "cv.shuffleI0.sci.b",
            "cv.shuffleI1.sci.b",
            "cv.shuffleI2.sci.b",
            "cv.shuffleI3.sci.b",
            "cv.shuffle2.h",
            "cv.shuffle2.b",
            "cv.pack",
            "cv.pack.h",
            "cv.packhi.b",
            "cv.packlo.b",
            "cv.cplxconj",
        ]

        for name in special_simd:
            simd_instrs[name] = (InstrFormat.XPULP_TYPE, InstrCategory.SIMD)

        # 带除法的 SIMD 操作
        div_variants = [".div2", ".div4", ".div8"]
        for div in div_variants:
            simd_instrs[f"cv.add{div}.h"] = (InstrFormat.XPULP_TYPE, InstrCategory.SIMD)
            simd_instrs[f"cv.sub{div}.h"] = (InstrFormat.XPULP_TYPE, InstrCategory.SIMD)
            simd_instrs[f"cv.subrotmj{div}"] = (
                InstrFormat.XPULP_TYPE,
                InstrCategory.SIMD,
            )
            simd_instrs[f"cv.cplxmul.r{div}"] = (
                InstrFormat.XPULP_TYPE,
                InstrCategory.SIMD,
            )
            simd_instrs[f"cv.cplxmul.i{div}"] = (
                InstrFormat.XPULP_TYPE,
                InstrCategory.SIMD,
            )

        simd_instrs["cv.subrotmj"] = (InstrFormat.XPULP_TYPE, InstrCategory.SIMD)
        simd_instrs["cv.cplxmul.r"] = (InstrFormat.XPULP_TYPE, InstrCategory.SIMD)
        simd_instrs["cv.cplxmul.i"] = (InstrFormat.XPULP_TYPE, InstrCategory.SIMD)

        return simd_instrs

    # 初始化 SIMD 指令
    SIMD_INSTRS = _init_simd_instructions.__func__(None)  # type: ignore

    @classmethod
    def register_extension(
        cls, name: str, instrs: Dict[str, Tuple[InstrFormat, InstrCategory]]
    ):
        """注册新的指令扩展"""
        cls._registry.update(instrs)

    @classmethod
    def classify(cls, mnemonic: str) -> Tuple[InstrFormat, InstrCategory]:
        """
        分类指令，返回（格式，类别）

        Args:
            mnemonic: 指令助记符（小写）

        Returns:
            (InstrFormat, InstrCategory) 元组
        """
        mnemonic_lower = mnemonic.lower()

        # 检查汇编指令
        if mnemonic_lower.startswith("."):
            return (InstrFormat.DIRECTIVE, InstrCategory.DIRECTIVE)

        # 按优先级查找各指令集
        for instr_set in [
            cls.RV32I_INSTRS,
            cls.RV32M_INSTRS,
            cls.RV32F_INSTRS,
            cls.RV32C_INSTRS,
            cls.XPULP_INSTRS,
            cls.SIMD_INSTRS,
            cls.PSEUDO_INSTRS,
            cls._registry,
        ]:
            if mnemonic_lower in instr_set:
                return instr_set[mnemonic_lower]

        # 未知指令
        return (InstrFormat.PSEUDO, InstrCategory.UNKNOWN)

    @classmethod
    def is_terminator(cls, mnemonic: str) -> bool:
        """
        判断是否为基本块终结指令

        终结指令包括：分支、跳转、系统指令（如 mret, wfi）
        """
        _, category = cls.classify(mnemonic)
        return category in (
            InstrCategory.BRANCH,
            InstrCategory.JUMP,
        )

    @classmethod
    def is_directive(cls, mnemonic: str) -> bool:
        """判断是否为汇编指令"""
        return mnemonic.lower().startswith(".")

    @classmethod
    def is_label(cls, text: str) -> bool:
        """判断是否为标签定义"""
        text = text.strip()
        # 标签格式: name: 或 数字: (局部标签)
        if text.endswith(":"):
            return True
        return False

    @classmethod
    def get_label_name(cls, text: str) -> Optional[str]:
        """从标签行提取标签名"""
        text = text.strip()
        if text.endswith(":"):
            return text[:-1].strip()
        return None
