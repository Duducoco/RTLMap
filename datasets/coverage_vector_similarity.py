#!/usr/bin/env python3
"""Positive coverage similarity for pair contrastive learning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

COVERAGE_VECTOR_TYPES: tuple[str, ...] = (
    "line",
    "condition",
    "toggle",
    "fsm",
    "branch",
)


@dataclass(frozen=True)
class CoverageGeometryTargets:
    """Per-type set measures used to supervise coverage-space geometry."""

    density_a: tuple[float, ...]
    density_b: tuple[float, ...]
    size_mask: tuple[bool, ...]
    jaccard: tuple[float, ...]
    iou_mask: tuple[bool, ...]


def _element_count(vectors: Mapping, vector_type: str) -> int:
    vector = vectors.get(vector_type, {})
    return int(vector.get("element_count", 0))


def total_coverage_vector_length(vectors: Mapping) -> int:
    """Return concatenated binary vector length in fixed coverage-type order."""
    return sum(
        _element_count(vectors, vector_type) for vector_type in COVERAGE_VECTOR_TYPES
    )


def covered_global_ids(vectors: Mapping) -> set[int]:
    """Build covered global ids for the concatenated coverage vector.

    Missing sparse items are explicit zeroes. Only items with ``label == 1`` become
    covered ids. Item ids are validated against each type's ``element_count``.
    """
    covered_by_type = _covered_ids_by_type(vectors)
    result: set[int] = set()
    offset = 0
    for vector_type in COVERAGE_VECTOR_TYPES:
        element_count = _element_count(vectors, vector_type)
        result.update(offset + item_id for item_id in covered_by_type[vector_type])
        offset += element_count
    return result


def _covered_ids_by_type(vectors: Mapping) -> dict[str, set[int]]:
    """Return covered local ids for each coverage type, validating sparse items."""
    result: dict[str, set[int]] = {}
    for vector_type in COVERAGE_VECTOR_TYPES:
        vector = vectors.get(vector_type, {})
        element_count = int(vector.get("element_count", 0))
        covered: set[int] = set()
        for item in vector.get("items", []):
            item_id = int(item["id"])
            if item_id < 0 or item_id >= element_count:
                raise ValueError(
                    f"coverage vector item id out of range: "
                    f"{vector_type}[{item_id}] with element_count={element_count}"
                )
            label = int(item.get("label", 1))
            if label == 1:
                covered.add(item_id)
            elif label != 0:
                raise ValueError(f"unsupported coverage vector label: {label}")
        result[vector_type] = covered
    return result


def compute_coverage_vector_positive_similarity(
    vectors_a: Mapping, vectors_b: Mapping
) -> float:
    """Compute positive-only Jaccard similarity between coverage reports.

    The score is the unweighted mean of per-coverage-type Jaccard scores:

    ``|Covered(a) ∩ Covered(b)| / |Covered(a) ∪ Covered(b)|``.

    Coverage types with no positive element in either report are omitted from the
    mean, so matching zeroes cannot dominate the score. If all active types are
    empty, the function returns ``0.0``; pair IoU supervision masks those types.
    """
    targets = compute_coverage_geometry_targets(vectors_a, vectors_b)
    scores = [
        score for score, active in zip(targets.jaccard, targets.iou_mask) if active
    ]
    return sum(scores) / len(scores) if scores else 0.0


def compute_coverage_geometry_targets(
    vectors_a: Mapping, vectors_b: Mapping
) -> CoverageGeometryTargets:
    """Return typed density and Jaccard targets without collapsing type structure."""
    covered_a = _covered_ids_by_type(vectors_a)
    covered_b = _covered_ids_by_type(vectors_b)
    density_a: list[float] = []
    density_b: list[float] = []
    size_mask: list[bool] = []
    jaccard: list[float] = []
    iou_mask: list[bool] = []

    for vector_type in COVERAGE_VECTOR_TYPES:
        count_a = _element_count(vectors_a, vector_type)
        count_b = _element_count(vectors_b, vector_type)
        if count_a != count_b:
            raise ValueError(
                f"coverage vector length mismatch for {vector_type}: "
                f"{count_a} != {count_b}"
            )

        exists = count_a > 0
        union = covered_a[vector_type] | covered_b[vector_type]
        density_a.append(len(covered_a[vector_type]) / count_a if exists else 0.0)
        density_b.append(len(covered_b[vector_type]) / count_a if exists else 0.0)
        size_mask.append(exists)
        jaccard.append(
            len(covered_a[vector_type] & covered_b[vector_type]) / len(union)
            if union
            else 0.0
        )
        iou_mask.append(bool(union))

    return CoverageGeometryTargets(
        density_a=tuple(density_a),
        density_b=tuple(density_b),
        size_mask=tuple(size_mask),
        jaccard=tuple(jaccard),
        iou_mask=tuple(iou_mask),
    )
