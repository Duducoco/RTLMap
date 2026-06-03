#!/usr/bin/env python3
"""覆盖率相似度计算 — 用于对比学习的对间 Jaccard 相似度"""

import math
from collections.abc import Sequence


def compute_rate_jaccard(cov_a: float, cov_b: float, cov_merge: float) -> float:
    """从聚合覆盖率反推 Jaccard 相似度

    利用集合运算：|A ∩ B| = cov_a + cov_b - cov_merge
    Jaccard = |A ∩ B| / |A ∪ B| = (cov_a + cov_b - cov_merge) / cov_merge

    Args:
        cov_a:     test_a 的覆盖率 ∈ [0, 1]
        cov_b:     test_b 的覆盖率 ∈ [0, 1]
        cov_merge: test_a ∪ test_b 合并覆盖率 ∈ [0, 1]

    Returns:
        Jaccard 相似度 ∈ [0.0, 1.0]，任一输入为 NaN 或 cov_merge=0 时返回 0.0
    """
    if math.isnan(cov_a) or math.isnan(cov_b) or math.isnan(cov_merge):
        return 0.0
    if cov_merge <= 0.0:
        return 0.0
    return max(0.0, min(1.0, (cov_a + cov_b - cov_merge) / cov_merge))


def compute_mean_rate_jaccard(
    cov_a: Sequence[float],
    cov_b: Sequence[float],
    cov_merge: Sequence[float],
) -> float:
    """对多种覆盖率分别反推 Jaccard 后取平均。

    NaN 表示该覆盖率类型未收集，跳过该项；如果没有任何有效覆盖率项，返回 0.0。
    """
    scores = []
    for a, b, merged in zip(cov_a, cov_b, cov_merge):
        if math.isnan(a) or math.isnan(b) or math.isnan(merged):
            continue
        scores.append(compute_rate_jaccard(a, b, merged))

    if not scores:
        return 0.0
    return sum(scores) / len(scores)
