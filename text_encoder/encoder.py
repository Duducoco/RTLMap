"""CodeBERT 指令文本编码器"""

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from copy import copy

import orjson
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .config import TextEncoderConfig
from .formatter import InstructionFormatter

logger = logging.getLogger(__name__)

# I/O 并行化的默认线程数（受限于 GIL，主要加速文件 I/O + orjson 解析）
_IO_WORKERS = min(8, (os.cpu_count() or 1) + 4)


def _load_and_format_asm(path: str) -> tuple[str, list[str]]:
    """读取 ASM JSON 并格式化节点文本（I/O + 解析并行化单元）

    纯函数，无副作用，线程安全。

    Args:
        path: ASM CDFG JSON 文件路径

    Returns:
        (path, texts) 元组，texts 为各节点格式化后的指令文本
    """
    with open(path, "rb") as f:
        asm = orjson.loads(f.read())

    node_ids = list(asm["nodes"].keys())
    texts = [
        InstructionFormatter.format_block(asm["nodes"][nid].get("instructions", []))
        for nid in node_ids
    ]
    return path, texts


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
        # 正交初始化：保持 CodeBERT 嵌入的几何结构（向量间角度和相对距离）
        # 此层不参与梯度优化，初始化即最终权重
        self.projection = nn.Linear(self.model.config.hidden_size, config.output_dim)
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(42)
        nn.init.orthogonal_(
            self.projection.weight,
            gain=math.sqrt(self.model.config.hidden_size / config.output_dim),
        )
        nn.init.zeros_(self.projection.bias)
        torch.random.set_rng_state(rng_state)
        self.projection.to(config.device)

        logger.info(
            "文本编码器就绪: %d → %d 维",
            self.model.config.hidden_size,
            config.output_dim,
        )

    def _pool_batch(self, batch_texts: list[str]) -> torch.Tensor:
        """单 batch tokenize + CodeBERT forward + pooling → [B, hidden_size]

        纯推理，不经过投影层。结果保留在 GPU 上。
        """
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
                return outputs.last_hidden_state[:, 0, :]

            # mean pooling: 仅对非 padding token 取均值
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            return (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(
                min=1e-9
            )

    def encode_texts_pooled(self, texts: list[str]) -> torch.Tensor:
        """批量编码文本列表 → [N, hidden_size] pooled 张量（投影前）

        用于缓存场景：缓存与投影层权重无关的 CodeBERT 原始输出。

        Args:
            texts: 待编码的文本列表

        Returns:
            [N, hidden_size] 张量（hidden_size=768 for CodeBERT）
        """
        all_pooled = []
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i : i + self.config.batch_size]
            pooled = self._pool_batch(batch_texts)
            all_pooled.append(pooled.cpu())
        return torch.cat(all_pooled, dim=0)

    def project(self, pooled: torch.Tensor) -> torch.Tensor:
        """将 pooled 输出通过投影层 → [N, output_dim]

        Args:
            pooled: [N, hidden_size] CodeBERT pooled 输出

        Returns:
            [N, output_dim] 投影后的张量
        """
        with torch.no_grad():
            return self.projection(pooled.to(self.config.device)).cpu()

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
            pooled = self._pool_batch(batch_texts)
            with torch.no_grad():
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
        _, texts = _load_and_format_asm(asm_json_path)

        logger.debug("编码 %d 个 ASM 节点: %s", len(texts), asm_json_path)
        return self.encode_texts(texts)

    def encode_asm_jsons_batch(
        self, asm_json_paths: list[str]
    ) -> dict[str, torch.Tensor]:
        """批量编码多个 ASM JSON 文件

        使用 ThreadPoolExecutor 并行加载 + 解析 + 格式化，
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

        # 并行 I/O：orjson 解析 + 文本格式化
        with ThreadPoolExecutor(max_workers=_IO_WORKERS) as pool:
            for path, texts in pool.map(_load_and_format_asm, asm_json_paths):
                file_slices.append((path, len(texts)))
                all_texts.extend(texts)

        if not all_texts:
            return {}

        logger.info(
            "批量编码 %d 个文件，共 %d 个节点", len(file_slices), len(all_texts)
        )
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

    def encode_texts_pooled(self, texts: list[str]) -> torch.Tensor:
        """批量编码文本列表 → [N, hidden_size] pooled 张量（投影前）

        单设备直接委托；多设备均匀切分后线程并行。
        """
        if not texts:
            return torch.empty(0, self._encoders[0].model.config.hidden_size)

        n_enc = len(self._encoders)
        if n_enc == 1:
            return self._encoders[0].encode_texts_pooled(texts)

        chunk_size = (len(texts) + n_enc - 1) // n_enc
        chunks = [texts[i * chunk_size : (i + 1) * chunk_size] for i in range(n_enc)]
        work = [(self._encoders[i], chunks[i]) for i in range(n_enc) if chunks[i]]

        def _encode_pooled(
            args: tuple[InstructionEncoder, list[str]],
        ) -> torch.Tensor:
            enc, chunk = args
            return enc.encode_texts_pooled(chunk)

        with ThreadPoolExecutor(max_workers=len(work)) as pool:
            results = list(pool.map(_encode_pooled, work))

        return torch.cat(results, dim=0)

    def project(self, pooled: torch.Tensor) -> torch.Tensor:
        """将 pooled 输出通过投影层（使用第一个编码器的投影层）"""
        return self._encoders[0].project(pooled)

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
        chunks = [texts[i * chunk_size : (i + 1) * chunk_size] for i in range(n_enc)]
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

        使用 ThreadPoolExecutor 并行加载 + 解析 + 格式化，
        合并所有文件的文本，调用并行 encode_texts 后按文件切分。
        """
        all_texts: list[str] = []
        file_slices: list[tuple[str, int]] = []

        # 并行 I/O：orjson 解析 + 文本格式化
        with ThreadPoolExecutor(max_workers=_IO_WORKERS) as pool:
            for path, texts in pool.map(_load_and_format_asm, asm_json_paths):
                file_slices.append((path, len(texts)))
                all_texts.extend(texts)

        if not all_texts:
            return {}

        logger.info(
            "批量编码 %d 个文件，共 %d 个节点（%d GPU）",
            len(file_slices),
            len(all_texts),
            len(self._encoders),
        )
        all_embeddings = self.encode_texts(all_texts)

        result: dict[str, torch.Tensor] = {}
        offset = 0
        for path, count in file_slices:
            result[path] = all_embeddings[offset : offset + count]
            offset += count

        return result
