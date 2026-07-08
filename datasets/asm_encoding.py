#!/usr/bin/env python3
"""ASM instruction encoding cache helpers."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from text_encoder import TextEncoderConfig

logger = logging.getLogger(__name__)


def compute_asm_cache_key(asm_path: str, config_fingerprint: str) -> str:
    h = hashlib.sha256()
    with open(asm_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    h.update(config_fingerprint.encode("utf-8"))
    return h.hexdigest()[:32]


def load_asm_encodings(
    asm_paths: list[str],
    *,
    cache_root: str | Path,
    text_encoder_config: "TextEncoderConfig | None",
) -> dict[str, torch.Tensor]:
    """Load or compute projected ASM instruction encodings."""
    if text_encoder_config is None or not asm_paths:
        logger.info("ASM 编码: 跳过（无编码器配置或无 ASM 文件）")
        return {}

    cache_dir = Path(cache_root) / "asm_encoding_cache"
    config_fp = text_encoder_config.cache_fingerprint
    asm_encodings: dict[str, torch.Tensor] = {}
    text_encoder = None

    def _get_encoder(*, projection_only: bool = False):
        nonlocal text_encoder
        if text_encoder is None:
            from text_encoder import MultiGPUInstructionEncoder

            text_encoder = MultiGPUInstructionEncoder(
                text_encoder_config,
                projection_only=projection_only,
            )
        return text_encoder

    uncached_paths: list[str] = []
    unique_paths = sorted(set(asm_paths))

    for asm_path in tqdm(unique_paths, desc="加载 ASM 编码缓存"):
        cache_key = compute_asm_cache_key(asm_path, config_fp)
        cache_file = cache_dir / f"{cache_key}.pt"
        if cache_file.exists():
            try:
                pooled = torch.load(cache_file, weights_only=True)
                asm_encodings[asm_path] = _get_encoder(
                    projection_only=True
                ).project(pooled)
                del pooled
                continue
            except Exception as e:
                logger.warning("缓存文件损坏，将重新编码: %s (%s)", cache_file, e)
        uncached_paths.append(asm_path)

    logger.info(
        "ASM 编码: %d 缓存命中, %d 待编码",
        len(asm_encodings),
        len(uncached_paths),
    )

    if uncached_paths:
        if text_encoder is not None and text_encoder._projection_only:
            text_encoder = None
        encoder = _get_encoder(projection_only=False)
        cache_dir.mkdir(parents=True, exist_ok=True)

        new_count = 0
        for path, pooled_tensor in encoder.encode_asm_jsons_pooled_iter(uncached_paths):
            cache_key = compute_asm_cache_key(path, config_fp)
            torch.save(pooled_tensor, cache_dir / f"{cache_key}.pt")
            asm_encodings[path] = encoder.project(pooled_tensor)
            del pooled_tensor
            new_count += 1

        logger.info("ASM 编码: 已缓存 %d 个新文件", new_count)

    logger.info("ASM 编码完成: %d 个编码结果", len(asm_encodings))
    return asm_encodings
