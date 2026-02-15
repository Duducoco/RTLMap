"""CodeBERT 指令文本编码器"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from copy import copy

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

from .config import TextEncoderConfig
from .formatter import InstructionFormatter

logger = logging.getLogger(__name__)


class InstructionEncoder:
    """使用 CodeBERT 将 ASM 指令文本编码为固定维度向量

    加载预训练 CodeBERT，冻结参数，通过线性投影层将 768 维输出
    映射到 output_dim 维。仅用于特征提取，不支持微调。
    """

    def __init__(self, config: TextEncoderConfig):
        self.config = config
        logger.info("加载文本编码器: %s", config.model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name)
        self.model.eval()
        self.model.to(config.device)

        # 冻结 CodeBERT 参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 768 → output_dim 线性投影
        self.projection = nn.Linear(
            self.model.config.hidden_size, config.output_dim
        )
        self.projection.to(config.device)

        logger.info(
            "文本编码器就绪: %d → %d 维",
            self.model.config.hidden_size,
            config.output_dim,
        )

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """批量编码文本列表 → [N, output_dim] 张量

        Args:
            texts: 待编码的文本列表

        Returns:
            [N, output_dim] 张量
        """
        all_embeddings = []
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i : i + self.config.batch_size]
            tokens = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            tokens = {k: v.to(self.config.device) for k, v in tokens.items()}

            with torch.no_grad():
                outputs = self.model(**tokens)

                if self.config.pooling == "cls":
                    pooled = outputs.last_hidden_state[:, 0, :]
                else:
                    # mean pooling: 仅对非 padding token 取均值
                    mask = tokens["attention_mask"].unsqueeze(-1).float()
                    pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

                projected = self.projection(pooled)
            all_embeddings.append(projected.cpu())

        return torch.cat(all_embeddings, dim=0)

    def encode_asm_json(self, asm_json_path: str) -> torch.Tensor:
        """编码 ASM JSON 文件中所有节点 → [M, output_dim] 张量

        节点顺序与 datamodule.py 中 list(asm["nodes"].keys()) 一致。

        Args:
            asm_json_path: ASM CDFG JSON 文件路径

        Returns:
            [M, output_dim] 张量
        """
        with open(asm_json_path, encoding="utf-8") as f:
            asm = json.load(f)

        node_ids = list(asm["nodes"].keys())
        texts = [
            InstructionFormatter.format_block(
                asm["nodes"][nid].get("instructions", [])
            )
            for nid in node_ids
        ]

        logger.debug("编码 %d 个 ASM 节点: %s", len(texts), asm_json_path)
        return self.encode_texts(texts)

    def encode_asm_jsons_batch(
        self, asm_json_paths: list[str]
    ) -> dict[str, torch.Tensor]:
        """批量编码多个 ASM JSON 文件

        合并所有文件的文本为一个大列表，一次性编码后按文件切分。
        相比逐文件调用 encode_asm_json()，减少 GPU kernel launch 开销，
        充分利用大 batch 提升吞吐。

        Args:
            asm_json_paths: ASM CDFG JSON 文件路径列表

        Returns:
            {asm_path: [M, output_dim] tensor} 映射
        """
        all_texts: list[str] = []
        # (path, node_count) 记录每个文件的节点数，用于切分
        file_slices: list[tuple[str, int]] = []

        for path in asm_json_paths:
            with open(path, encoding="utf-8") as f:
                asm = json.load(f)

            node_ids = list(asm["nodes"].keys())
            texts = [
                InstructionFormatter.format_block(
                    asm["nodes"][nid].get("instructions", [])
                )
                for nid in node_ids
            ]
            file_slices.append((path, len(texts)))
            all_texts.extend(texts)

        if not all_texts:
            return {}

        logger.info("批量编码 %d 个文件，共 %d 个节点", len(file_slices), len(all_texts))
        all_embeddings = self.encode_texts(all_texts)

        # 按文件切分
        result: dict[str, torch.Tensor] = {}
        offset = 0
        for path, count in file_slices:
            result[path] = all_embeddings[offset : offset + count]
            offset += count

        return result


class MultiGPUInstructionEncoder:
    """多 GPU 并行编码器

    内部持有 N 个 InstructionEncoder 实例（每个绑定不同 GPU），
    文本均匀切分后用 ThreadPoolExecutor 并行推理。
    CUDA 操作在 C++ 层释放 GIL，线程可真正并行。

    单设备时直接委托给内部唯一的 InstructionEncoder，零开销。
    """

    def __init__(self, config: TextEncoderConfig):
        # 自动检测可用 GPU：有几张用几张，没有则回退 CPU
        gpu_count = torch.cuda.device_count()
        if gpu_count > 0:
            devices = [f"cuda:{i}" for i in range(gpu_count)]
        else:
            devices = ["cpu"]

        logger.info("初始化多 GPU 编码器: %s", devices)

        self._encoders: list[InstructionEncoder] = []
        for dev in devices:
            per_dev_config = copy(config)
            per_dev_config.device = dev
            self._encoders.append(InstructionEncoder(per_dev_config))

        logger.info("多 GPU 编码器就绪: %d 个设备", len(self._encoders))

    # --- PLACEHOLDER_ENCODE_TEXTS ---

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """批量编码文本列表 → [N, output_dim] 张量

        单设备直接委托；多设备均匀切分后线程并行。
        """
        if not texts:
            return torch.empty(0, self._encoders[0].config.output_dim)

        n_enc = len(self._encoders)
        if n_enc == 1:
            return self._encoders[0].encode_texts(texts)

        # 均匀切分
        chunk_size = (len(texts) + n_enc - 1) // n_enc
        chunks = [
            texts[i * chunk_size : (i + 1) * chunk_size]
            for i in range(n_enc)
        ]
        # 过滤空 chunk（文本数 < 设备数时）
        work = [(self._encoders[i], chunks[i]) for i in range(n_enc) if chunks[i]]

        def _encode(args: tuple[InstructionEncoder, list[str]]) -> torch.Tensor:
            enc, chunk = args
            return enc.encode_texts(chunk)

        with ThreadPoolExecutor(max_workers=len(work)) as pool:
            results = list(pool.map(_encode, work))

        return torch.cat(results, dim=0)

    # --- PLACEHOLDER_BATCH ---

    def encode_asm_jsons_batch(
        self, asm_json_paths: list[str]
    ) -> dict[str, torch.Tensor]:
        """批量编码多个 ASM JSON 文件

        合并所有文件的文本，调用并行 encode_texts 后按文件切分。
        """
        all_texts: list[str] = []
        file_slices: list[tuple[str, int]] = []

        for path in asm_json_paths:
            with open(path, encoding="utf-8") as f:
                asm = json.load(f)

            node_ids = list(asm["nodes"].keys())
            texts = [
                InstructionFormatter.format_block(
                    asm["nodes"][nid].get("instructions", [])
                )
                for nid in node_ids
            ]
            file_slices.append((path, len(texts)))
            all_texts.extend(texts)

        if not all_texts:
            return {}

        logger.info("批量编码 %d 个文件，共 %d 个节点（%d GPU）",
                     len(file_slices), len(all_texts), len(self._encoders))
        all_embeddings = self.encode_texts(all_texts)

        result: dict[str, torch.Tensor] = {}
        offset = 0
        for path, count in file_slices:
            result[path] = all_embeddings[offset : offset + count]
            offset += count

        return result
