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


@dataclass(frozen=True)
class CoverageSignature:
    """Validated, compact coverage representation for repeated pair scoring."""

    element_counts: tuple[int, ...]
    covered_masks: tuple[int, ...]
    covered_counts: tuple[int, ...]


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


def build_coverage_signature(vectors: Mapping) -> CoverageSignature:
    """Validate sparse vectors and encode covered ids as integer bitmasks."""
    element_counts: list[int] = []
    covered_masks: list[int] = []
    covered_counts: list[int] = []
    for vector_type in COVERAGE_VECTOR_TYPES:
        vector = vectors.get(vector_type, {})
        element_count = int(vector.get("element_count", 0))
        mask = 0
        for item in vector.get("items", []):
            item_id = int(item["id"])
            if item_id < 0 or item_id >= element_count:
                raise ValueError(
                    f"coverage vector item id out of range: "
                    f"{vector_type}[{item_id}] with element_count={element_count}"
                )
            label = int(item.get("label", 1))
            if label == 1:
                mask |= 1 << item_id
            elif label != 0:
                raise ValueError(f"unsupported coverage vector label: {label}")
        element_counts.append(element_count)
        covered_masks.append(mask)
        covered_counts.append(mask.bit_count())
    return CoverageSignature(
        element_counts=tuple(element_counts),
        covered_masks=tuple(covered_masks),
        covered_counts=tuple(covered_counts),
    )


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


def coverage_similarity_from_signatures(
    signature_a: CoverageSignature, signature_b: CoverageSignature
) -> float:
    """Compute positive-only Jaccard similarity from validated signatures."""
    scores: list[float] = []
    for count_a, count_b, mask_a, mask_b in zip(
        signature_a.element_counts,
        signature_b.element_counts,
        signature_a.covered_masks,
        signature_b.covered_masks,
        strict=True,
    ):
        if count_a != count_b:
            raise ValueError(f"coverage vector length mismatch: {count_a} != {count_b}")
        union = mask_a | mask_b
        if union:
            scores.append((mask_a & mask_b).bit_count() / union.bit_count())
    return sum(scores) / len(scores) if scores else 0.0


def compute_coverage_geometry_targets_from_signatures(
    signature_a: CoverageSignature, signature_b: CoverageSignature
) -> CoverageGeometryTargets:
    """Compute geometry targets from validated coverage signatures."""
    density_a: list[float] = []
    density_b: list[float] = []
    size_mask: list[bool] = []
    jaccard: list[float] = []
    iou_mask: list[bool] = []
    for count_a, count_b, mask_a, mask_b, covered_a, covered_b in zip(
        signature_a.element_counts,
        signature_b.element_counts,
        signature_a.covered_masks,
        signature_b.covered_masks,
        signature_a.covered_counts,
        signature_b.covered_counts,
        strict=True,
    ):
        if count_a != count_b:
            raise ValueError(f"coverage vector length mismatch: {count_a} != {count_b}")
        exists = count_a > 0
        union = mask_a | mask_b
        union_count = union.bit_count()
        density_a.append(covered_a / count_a if exists else 0.0)
        density_b.append(covered_b / count_a if exists else 0.0)
        size_mask.append(exists)
        jaccard.append(
            (mask_a & mask_b).bit_count() / union_count if union_count else 0.0
        )
        iou_mask.append(bool(union))
    return CoverageGeometryTargets(
        density_a=tuple(density_a),
        density_b=tuple(density_b),
        size_mask=tuple(size_mask),
        jaccard=tuple(jaccard),
        iou_mask=tuple(iou_mask),
    )


def compute_coverage_geometry_targets(
    vectors_a: Mapping, vectors_b: Mapping
) -> CoverageGeometryTargets:
    """Return typed density and Jaccard targets without collapsing type structure."""
    return compute_coverage_geometry_targets_from_signatures(
        build_coverage_signature(vectors_a), build_coverage_signature(vectors_b)
    )
