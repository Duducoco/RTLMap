"""文本编码器配置"""

from dataclasses import dataclass


@dataclass
class TextEncoderConfig:
    """CodeBERT 文本编码器配置

    Attributes:
        model_name: HuggingFace 模型名称
        output_dim: 投影后维度，匹配 ModelConfig.asm_instruction_dim
        max_length: tokenizer 最大长度（CodeBERT 最大支持 512）
        batch_size: 编码批大小
        device: 编码设备
        pooling: token 级池化策略: "mean" | "cls"
    """

    model_name: str = "microsoft/codebert-base"
    output_dim: int = 256
    max_length: int = 512
    batch_size: int = 512
    device: str = "cpu"
    pooling: str = "mean"

    @property
    def cache_fingerprint(self) -> str:
        """影响 CodeBERT 768 维 pooled 输出的参数指纹

        output_dim / batch_size / device 不影响 pooled 输出，不纳入指纹。
        """
        return f"{self.model_name}|{self.max_length}|{self.pooling}"
