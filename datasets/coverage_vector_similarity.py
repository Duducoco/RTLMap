#!/usr/bin/env python3
"""Coverage-vector agreement similarity for pair contrastive learning."""

from __future__ import annotations

from collections.abc import Mapping

COVERAGE_VECTOR_TYPES: tuple[str, ...] = (
    "line",
    "condition",
    "toggle",
    "fsm",
    "branch",
)


def _element_count(vectors: Mapping, vector_type: str) -> int:
    vector = vectors.get(vector_type, {})
    return int(vector.get("element_count", 0))


def total_coverage_vector_length(vectors: Mapping) -> int:
    """Return concatenated binary vector length in fixed coverage-type order."""
    return sum(_element_count(vectors, vector_type) for vector_type in COVERAGE_VECTOR_TYPES)


def covered_global_ids(vectors: Mapping) -> set[int]:
    """Build covered global ids for the concatenated coverage vector.

    Missing sparse items are explicit zeroes. Only items with ``label == 1`` become
    covered ids. Item ids are validated against each type's ``element_count``.
    """
    result: set[int] = set()
    offset = 0
    for vector_type in COVERAGE_VECTOR_TYPES:
        vector = vectors.get(vector_type, {})
        element_count = int(vector.get("element_count", 0))
        for item in vector.get("items", []):
            item_id = int(item["id"])
            if item_id < 0 or item_id >= element_count:
                raise ValueError(
                    f"coverage vector item id out of range: "
                    f"{vector_type}[{item_id}] with element_count={element_count}"
                )
            label = int(item.get("label", 1))
            if label == 1:
                result.add(offset + item_id)
            elif label != 0:
                raise ValueError(f"unsupported coverage vector label: {label}")
        offset += element_count
    return result


def compute_coverage_vector_agreement(vectors_a: Mapping, vectors_b: Mapping) -> float:
    """Compute full binary-vector agreement between two coverage reports.

    similarity = 1 - |Covered(a) symmetric_difference Covered(b)| / N

    where N is the concatenated vector length. Matching zeroes and matching ones
    both increase similarity.
    """
    length_a = total_coverage_vector_length(vectors_a)
    length_b = total_coverage_vector_length(vectors_b)
    if length_a != length_b:
        raise ValueError(
            f"coverage vector length mismatch: {length_a} != {length_b}"
        )
    if length_a <= 0:
        raise ValueError("coverage vector length is zero")

    covered_a = covered_global_ids(vectors_a)
    covered_b = covered_global_ids(vectors_b)
    return 1.0 - (len(covered_a.symmetric_difference(covered_b)) / length_a)
