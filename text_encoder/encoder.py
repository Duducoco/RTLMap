"""CodeBERT 指令文本编码器"""

import json
import logging
import math
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from copy import copy
import re
import orjson
import torch
from tqdm import tqdm
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .config import TextEncoderConfig
from .formatter import InstructionFormatter

logger = logging.getLogger(__name__)

# I/O 并行化的默认线程数
_IO_WORKERS = os.cpu_count() or 1


def _clean_json_control_chars(raw: bytes) -> bytes:
    """清理 JSON 中的非法控制字符

    JSON 规范要求字符串内的控制字符（U+0000 到 U+001F）必须被转义。
    此函数将非法控制字符替换为空格，保留合法的空白字符（制表符、换行符、回车符）。

    使用空格替换而非删除，以保持 JSON 结构完整性（避免破坏字符串边界）。

    Args:
        raw: 原始 JSON 字节数据

    Returns:
        清理后的 JSON 字节数据
    """
    # 将控制字符替换为空格，除了：
    # - \x09 (制表符 \t)
    # - \x0a (换行符 \n)
    # - \x0d (回车符 \r)
    # 同时处理 DEL 字符 \x7f 和扩展控制字符 \x80-\x9f
    return re.sub(rb"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", b" ", raw)


def _parse_json_lenient(raw: bytes, path: str) -> dict:
    """宽松模式解析 JSON，处理各种非法字符

    尝试顺序：
    1. orjson 直接解析（最快）
    2. 清理控制字符后 orjson 解析
    3. Python json 模块 strict=False 模式（最宽松）

    Args:
        raw: 原始 JSON 字节数据
        path: 文件路径（用于错误日志）

    Returns:
        解析后的字典

    Raises:
        json.JSONDecodeError: 所有方法都失败时抛出
    """
    # 方法1：直接尝试 orjson
    try:
        return orjson.loads(raw)
    except orjson.JSONDecodeError:
        pass

    # 方法2：清理控制字符后重试 orjson
    cleaned = _clean_json_control_chars(raw)
    try:
        return orjson.loads(cleaned)
    except orjson.JSONDecodeError:
        pass

    # 方法3：使用 Python json 模块的宽松模式
    # strict=False 允许字符串中包含控制字符
    try:
        # 先解码为字符串，忽略无法解码的字节
        text = cleaned.decode("utf-8", errors="replace")
        return json.loads(text, strict=False)
    except json.JSONDecodeError as e:
        logger.error("JSON 解析失败（所有方法）: %s, 错误: %s", path, str(e))
        raise


def _load_and_format_asm(path: str) -> tuple[str, list[str]]:
    """读取 ASM JSON 并格式化节点文本（I/O + 解析并行化单元）

    纯函数，无副作用，线程安全。

    Args:
        path: ASM CDFG JSON 文件路径

    Returns:
        (path, texts) 元组，texts 为各节点格式化后的指令文本
    """
    with open(path, "rb") as f:
        raw = f.read()
    try:
        asm = _parse_json_lenient(raw, path)
    except (json.JSONDecodeError, orjson.JSONDecodeError):
        # 损坏的 JSON 已在 _parse_json_lenient 中记录 ERROR，跳过该文件
        return path, []
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

    Args:
        config: 文本编码器配置
        projection_only: 仅初始化投影层，跳过 CodeBERT 模型和 tokenizer 加载。
            用于全缓存命中场景，仅需投影功能时节省约 500MB×GPU数 的显存。
    """

    # CodeBERT 固定隐藏维度
    HIDDEN_SIZE = 768

    def __init__(self, config: TextEncoderConfig, *, projection_only: bool = False):
        self.config = config
        self._projection_only = projection_only

        if projection_only:
            logger.info("仅加载投影层: %d → %d 维", self.HIDDEN_SIZE, config.output_dim)
            self.tokenizer = None
            self.model = None
            hidden_size = self.HIDDEN_SIZE
        else:
            logger.info("加载文本编码器: %s", config.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model = AutoModel.from_pretrained(config.model_name)
            self.model.eval()
            self.model.to(config.device)

            # 冻结 CodeBERT 参数
            for param in self.model.parameters():
                param.requires_grad = False

            hidden_size = self.model.config.hidden_size

        # 768 → output_dim 线性投影
        # 正交初始化：保持 CodeBERT 嵌入的几何结构（向量间角度和相对距离）
        # 此层不参与梯度优化，初始化即最终权重
        self.projection = nn.Linear(hidden_size, config.output_dim)
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(42)
        nn.init.orthogonal_(
            self.projection.weight,
            gain=math.sqrt(hidden_size / config.output_dim),
        )
        nn.init.zeros_(self.projection.bias)
        torch.random.set_rng_state(rng_state)
        self.projection.to(config.device)

        logger.info(
            "文本编码器就绪: %d → %d 维%s",
            hidden_size,
            config.output_dim,
            "（仅投影）" if projection_only else "",
        )

    def _tokenize_batch(self, batch_texts: list[str]) -> dict[str, torch.Tensor]:
        """CPU tokenize + pin_memory，为非阻塞 GPU 传输做准备"""
        tokens = self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        # pin_memory 允许 .to(device, non_blocking=True) 异步传输
        return {k: v.pin_memory() for k, v in tokens.items()}

    def _pool_batch(self, batch_texts: list[str]) -> torch.Tensor:
        """单 batch tokenize + CodeBERT forward + pooling → [B, hidden_size]

        纯推理，不经过投影层。结果保留在 GPU 上。
        """
        tokens = self._tokenize_batch(batch_texts)
        return self._forward_tokens(tokens)

    def _forward_tokens(self, tokens: dict[str, torch.Tensor]) -> torch.Tensor:
        """已 tokenize 的输入 → GPU forward + pooling → [B, hidden_size]"""
        tokens = {
            k: v.to(self.config.device, non_blocking=True) for k, v in tokens.items()
        }

        with torch.no_grad():
            outputs = self.model(**tokens)

            if self.config.pooling == "cls":
                return outputs.last_hidden_state[:, 0, :]

            # mean pooling: 仅对非 padding token 取均值
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            return (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(
                min=1e-9
            )

    def _pool_batch_safe(self, batch_texts: list[str]) -> torch.Tensor:
        """带 OOM 自动降级的 batch 编码

        OOM 时清理显存并将 batch 减半重试，递归直到成功。
        中间结果立即移至 CPU，避免递归降级时显存累积。
        """
        try:
            return self._pool_batch(batch_texts).cpu()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            half = len(batch_texts) // 2
            if half == 0:
                raise  # 单条文本仍 OOM，无法降级
            logger.warning("CUDA OOM，batch %d → %d 重试", len(batch_texts), half)
            left = self._pool_batch_safe(batch_texts[:half])  # 已在 CPU
            right = self._pool_batch_safe(batch_texts[half:])  # 已在 CPU
            return torch.cat([left, right], dim=0)

    def encode_texts_pooled(self, texts: list[str]) -> torch.Tensor:
        """批量编码文本列表 → [N, hidden_size] pooled 张量（投影前）

        流水线：预取线程 tokenize batch N+1，同时 GPU 推理 batch N。

        Args:
            texts: 待编码的文本列表

        Returns:
            [N, hidden_size] 张量（hidden_size=768 for CodeBERT）
        """
        if self._projection_only:
            raise RuntimeError("projection_only 模式下不支持编码操作")
        if not texts:
            return torch.empty(0, self.model.config.hidden_size)

        bs = self.config.batch_size
        batches = [texts[i : i + bs] for i in range(0, len(texts), bs)]

        all_pooled = []
        # 预取线程：CPU tokenize 与 GPU forward 流水线并行
        prefetch_pool = ThreadPoolExecutor(max_workers=1)
        next_tokens = prefetch_pool.submit(self._tokenize_batch, batches[0])

        for i in tqdm(range(len(batches)), desc="CodeBERT 编码"):
            tokens = next_tokens.result()
            # 提前提交下一个 batch 的 tokenize
            if i + 1 < len(batches):
                next_tokens = prefetch_pool.submit(self._tokenize_batch, batches[i + 1])
            try:
                pooled = self._forward_tokens(tokens)
            except torch.cuda.OutOfMemoryError:
                # OOM 降级：回退到 _pool_batch_safe 逐条处理
                torch.cuda.empty_cache()
                logger.warning("CUDA OOM，batch %d 降级处理", len(batches[i]))
                pooled = self._pool_batch_safe(batches[i])
            all_pooled.append(pooled.cpu())

        prefetch_pool.shutdown(wait=False)
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


class MultiGPUInstructionEncoder:
    """多 GPU 并行编码器

    内部持有 N 个 InstructionEncoder 实例（每个绑定不同 GPU），
    文本均匀切分后用 ThreadPoolExecutor 并行推理。
    CUDA 操作在 C++ 层释放 GIL，线程可真正并行。

    单设备时直接委托给内部唯一的 InstructionEncoder，零开销。
    """

    def __init__(self, config: TextEncoderConfig, *, projection_only: bool = False):
        self._projection_only = projection_only

        if projection_only:
            # 仅投影模式：单实例即可，无需多 GPU
            logger.info("初始化多 GPU 编码器（仅投影模式）")
            per_dev_config = copy(config)
            per_dev_config.device = "cpu"
            self._encoders = [InstructionEncoder(per_dev_config, projection_only=True)]
        else:
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

    def encode_texts_pooled(self, texts: list[str]) -> torch.Tensor:
        """批量编码文本列表 → [N, hidden_size] pooled 张量（投影前）

        单设备直接委托；多设备均匀切分后线程并行。
        """
        if self._projection_only:
            raise RuntimeError("projection_only 模式下不支持编码操作")
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

    def encode_asm_jsons_pooled_iter(
        self, asm_json_paths: list[str], *, chunk_files: int = 8
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """逐文件编码并 yield (path, pooled_tensor)

        分块惰性加载 + 分块批量编码 + 逐文件 yield。
        每个 chunk 编码完即可持久化，中断后只需重编未完成的文件。
        文本内存从 O(所有未缓存文件) 降到 O(chunk_files)。

        Args:
            asm_json_paths: ASM CDFG JSON 文件路径列表
            chunk_files: 每次编码的文件数，平衡 GPU 吞吐与中断粒度

        Yields:
            (asm_path, [M, hidden_size] tensor) 元组
        """
        if not asm_json_paths:
            return

        total_files = len(asm_json_paths)
        total_nodes = 0

        # 按 chunk_files 分批路径，每批独立加载文本 + 编码
        for chunk_start in range(0, total_files, chunk_files):
            chunk_paths = asm_json_paths[chunk_start : chunk_start + chunk_files]

            # 仅加载当前批次的文本（避免一次性预加载所有文件）
            chunk_data: list[tuple[str, list[str]]] = []
            with ProcessPoolExecutor(max_workers=_IO_WORKERS) as pool:
                results = pool.map(_load_and_format_asm, chunk_paths)
                for path, texts in results:
                    if texts:
                        chunk_data.append((path, texts))

            if not chunk_data:
                continue

            chunk_nodes = sum(len(texts) for _, texts in chunk_data)
            total_nodes += chunk_nodes

            if chunk_start == 0:
                logger.info(
                    "批量编码 %d 个文件（%d GPU，每 %d 文件一批）",
                    total_files,
                    len(self._encoders),
                    chunk_files,
                )

            # 编码当前批次
            chunk_texts: list[str] = []
            chunk_slices: list[tuple[str, int]] = []
            for path, texts in chunk_data:
                chunk_slices.append((path, len(texts)))
                chunk_texts.extend(texts)

            chunk_pooled = self.encode_texts_pooled(chunk_texts)

            offset = 0
            for path, count in chunk_slices:
                yield path, chunk_pooled[offset : offset + count]
                offset += count

            # 释放当前批次的文本和编码结果
            del chunk_data, chunk_texts, chunk_pooled

        logger.info("批量编码完成，共 %d 个节点", total_nodes)
