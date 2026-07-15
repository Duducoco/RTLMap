#!/usr/bin/env python3
"""Tests for the coverage-report-extractor manifest dataset format."""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import torch
import lightning as L

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import DualGraphDataset
from datasets.coverage_vector_similarity import (
    build_coverage_signature,
    compute_coverage_geometry_targets,
    compute_coverage_geometry_targets_from_signatures,
    compute_coverage_vector_positive_similarity,
    coverage_similarity_from_signatures,
)
from datasets.pair_datamodule import ContrastivePairDataset
from models import ModelConfig
from trainer.config import TrainerConfig
from trainer.lightning_module import DualGraphLightningModule
from trainer.utils import _build_training_datamodule


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _minimal_rtl_graph() -> dict:
    return {
        "schema_version": "rtl_graph.v1",
        "module_name": "ALU",
        "nodes": [
            {
                "id": "n0",
                "type": "INPUT",
                "cell_type": "INPUT",
                "width": 1,
                "input_ports": [],
                "output_ports": ["Y"],
            },
            {
                "id": "n1",
                "type": "OUTPUT",
                "cell_type": "OUTPUT",
                "width": 1,
                "input_ports": ["A"],
                "output_ports": [],
            },
        ],
        "edges": [
            {
                "edge_id": 0,
                "source": "missing",
                "target": "n1",
                "source_port_idx": 0,
                "target_port_idx": 0,
                "type": "DATA",
                "width": 1,
            },
            {
                "edge_id": 1,
                "source": "n0",
                "target": "n1",
                "source_port_idx": 0,
                "target_port_idx": 0,
                "type": "DATA_TRUE",
                "width": 1,
            },
        ],
    }


def _minimal_asm_graph() -> dict:
    return {
        "entry_node": "bb_0",
        "exit_nodes": ["bb_0"],
        "nodes": {
            "bb_0": {
                "id": "bb_0",
                "node_type": "ARITHMETIC",
                "instructions": [{"mnemonic": "add", "operands": ["x1", "x2", "x3"]}],
            }
        },
        "edges": [],
    }


def _sample_entry(sample_id: str, asm_graph: str | None) -> dict:
    return {
        "schema_version": "sample.v1",
        "sample_id": sample_id,
        "test_id": sample_id.split("::")[0],
        "module_name": "ALU",
        "asm_graph": asm_graph,
        "rtl_graph": "rtl_graphs/ALU.json",
        "rtl_graph_hash": "sha256:test",
        "asm_graph_hash": None,
        "targets": {
            "edge_coverage": {
                "encoding": {"-1": "ignore", "0": "uncovered", "1": "covered"},
                "edge_count": 2,
                "default": -1,
                "labels": [{"edge_id": 1, "label": 0}],
            },
            "graph_coverage": {
                "unit": "percent",
                "scope": "module",
                "values": {
                    "branch": 50.0,
                    "line": None,
                    "fsm": 0.0,
                    "toggle": 100.0,
                    "condition": None,
                },
                "mask": {
                    "branch": True,
                    "line": False,
                    "fsm": True,
                    "toggle": True,
                    "condition": False,
                },
            },
        },
    }


def _coverage_vectors(
    *,
    line_count: int = 2,
    condition_count: int = 0,
    toggle_count: int = 0,
    fsm_count: int = 0,
    branch_count: int = 3,
    line_ids: list[int] | None = None,
    condition_ids: list[int] | None = None,
    toggle_ids: list[int] | None = None,
    fsm_ids: list[int] | None = None,
    branch_ids: list[int] | None = None,
) -> dict:
    def vector(name: str, count: int, ids: list[int] | None) -> dict:
        return {
            "schema": "coverage_schemas/ALU.json",
            "element_count": count,
            "items": [{"id": item_id, "label": 1} for item_id in (ids or [])],
        }

    return {
        "line": vector("line", line_count, line_ids),
        "condition": vector("condition", condition_count, condition_ids),
        "toggle": vector("toggle", toggle_count, toggle_ids),
        "fsm": vector("fsm", fsm_count, fsm_ids),
        "branch": vector("branch", branch_count, branch_ids),
    }


def _sample_entry_with_vectors(
    sample_id: str,
    asm_graph: str | None,
    coverage_vectors: dict,
) -> dict:
    sample = _sample_entry(sample_id, asm_graph)
    sample["targets"]["coverage_vectors"] = coverage_vectors
    return sample


def test_coverage_vector_positive_similarity_uses_jaccard() -> None:
    vec_a = _coverage_vectors(line_ids=[0], branch_ids=[0])
    vec_b = _coverage_vectors(line_ids=[0], branch_ids=[1])

    similarity = compute_coverage_vector_positive_similarity(vec_a, vec_b)

    # line has Jaccard 1.0; branch has Jaccard 0.0. Other empty types are omitted.
    assert math.isclose(similarity, 0.5)


def test_coverage_signature_preserves_geometry_targets() -> None:
    vec_a = _coverage_vectors(line_count=3, line_ids=[0, 1], branch_ids=[0])
    vec_b = _coverage_vectors(line_count=3, line_ids=[1, 2], branch_ids=[1])

    signature_a = build_coverage_signature(vec_a)
    signature_b = build_coverage_signature(vec_b)

    expected = compute_coverage_geometry_targets(vec_a, vec_b)
    actual = compute_coverage_geometry_targets_from_signatures(signature_a, signature_b)

    assert actual == expected
    assert coverage_similarity_from_signatures(signature_a, signature_b) == (
        compute_coverage_vector_positive_similarity(vec_a, vec_b)
    )


def test_coverage_vector_positive_similarity_does_not_count_matching_zeros() -> None:
    vec_a = _coverage_vectors(line_ids=[0], branch_ids=[0])
    vec_b = _coverage_vectors(line_ids=[1], branch_ids=[1])

    assert compute_coverage_vector_positive_similarity(vec_a, vec_b) == 0.0


def test_coverage_vector_positive_similarity_returns_zero_for_two_empty_reports() -> (
    None
):
    vec_a = _coverage_vectors()
    vec_b = _coverage_vectors()

    assert compute_coverage_vector_positive_similarity(vec_a, vec_b) == 0.0


def test_coverage_vector_positive_similarity_rejects_out_of_range_item_id() -> None:
    vec_a = _coverage_vectors(line_count=2, line_ids=[2])
    vec_b = _coverage_vectors(line_count=2, line_ids=[])

    try:
        compute_coverage_vector_positive_similarity(vec_a, vec_b)
    except ValueError as exc:
        assert "out of range" in str(exc)
    else:
        raise AssertionError("expected out-of-range coverage vector id to fail")


def test_pair_contrastive_dataset_uses_dataset_v1_coverage_vectors() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        dataset_dir = base / "dataset"
        _write_json(dataset_dir / "rtl_graphs" / "ALU.json", _minimal_rtl_graph())
        _write_json(dataset_dir / "asm_graphs" / "test_a.json", _minimal_asm_graph())
        _write_json(
            dataset_dir / "manifest.json",
            {
                "schema_version": "dataset.v1",
                "dataset_name": "unit",
                "samples": "samples.jsonlines",
                "rtl_graphs": "rtl_graphs",
                "asm_graphs": "asm_graphs",
                "total": 2,
                "errors": 0,
            },
        )
        samples = [
            _sample_entry_with_vectors(
                "test_a::ALU",
                "asm_graphs/test_a.json",
                _coverage_vectors(line_ids=[0], branch_ids=[0]),
            ),
            _sample_entry_with_vectors(
                "test_b::ALU",
                None,
                _coverage_vectors(line_ids=[0], branch_ids=[1]),
            ),
        ]
        (dataset_dir / "samples.jsonlines").write_text(
            "".join(json.dumps(s) + "\n" for s in samples),
            encoding="utf-8",
        )

        cache_root = base / "cache" / "train"
        prepared = DualGraphDataset(
            root=str(cache_root),
            dataset_dir=dataset_dir,
            num_workers=1,
        )
        assert len(prepared) == 2

        import datasets.datamodule as datamodule_module

        original_load_asm_encodings = datamodule_module._load_asm_encodings

        def fail_if_asm_cache_loads(*args, **kwargs):
            del args, kwargs
            raise AssertionError("pair training should read processed graph cache")

        datamodule_module._load_asm_encodings = fail_if_asm_cache_loads
        try:
            dataset = ContrastivePairDataset(
                dataset_dir=dataset_dir,
                text_encoder_config=object(),
                processed_root=cache_root,
            )
        finally:
            datamodule_module._load_asm_encodings = original_load_asm_encodings

        assert len(dataset) == 1
        data_a, data_b, targets, weight_a, weight_b = dataset[0]
        assert data_a.edge_labels.tolist() == [0]
        assert data_b.edge_labels.tolist() == [0]
        assert targets.density_a == (0.5, 0.0, 0.0, 0.0, 1 / 3)
        assert targets.density_b == (0.5, 0.0, 0.0, 0.0, 1 / 3)
        assert targets.size_mask == (True, False, False, False, True)
        assert targets.jaccard == (1.0, 0.0, 0.0, 0.0, 0.0)
        assert targets.iou_mask == (True, False, False, False, True)
        assert weight_a == 1.0
        assert weight_b == 1.0


def test_pair_contrastive_dataset_does_not_pair_same_module_across_dirs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        dataset_dirs = []
        for idx, covered_branch in enumerate([0, 1]):
            dataset_dir = base / f"dataset_{idx}"
            dataset_dirs.append(dataset_dir)
            _write_json(dataset_dir / "rtl_graphs" / "ALU.json", _minimal_rtl_graph())
            _write_json(
                dataset_dir / "manifest.json",
                {
                    "schema_version": "dataset.v1",
                    "dataset_name": f"unit_{idx}",
                    "samples": "samples.jsonlines",
                    "rtl_graphs": "rtl_graphs",
                    "asm_graphs": "asm_graphs",
                    "total": 1,
                    "errors": 0,
                },
            )
            sample = _sample_entry_with_vectors(
                f"test_{idx}::ALU",
                None,
                _coverage_vectors(line_ids=[0], branch_ids=[covered_branch]),
            )
            (dataset_dir / "samples.jsonlines").write_text(
                json.dumps(sample) + "\n",
                encoding="utf-8",
            )

        dataset = ContrastivePairDataset(
            dataset_dir=dataset_dirs,
            processed_root=base / "cache" / "train",
        )

        assert len(dataset) == 0


def test_manifest_dataset_loads_sparse_targets_without_legacy_index() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        dataset_dir = base / "dataset"
        _write_json(dataset_dir / "rtl_graphs" / "ALU.json", _minimal_rtl_graph())
        _write_json(dataset_dir / "asm_graphs" / "test_a.json", _minimal_asm_graph())
        _write_json(
            dataset_dir / "manifest.json",
            {
                "schema_version": "dataset.v1",
                "dataset_name": "unit",
                "samples": "samples.jsonlines",
                "rtl_graphs": "rtl_graphs",
                "asm_graphs": "asm_graphs",
                "total": 2,
                "errors": 0,
            },
        )
        samples = [
            _sample_entry("test_a::ALU", "asm_graphs/test_a.json"),
            _sample_entry("test_b::ALU", None),
        ]
        (dataset_dir / "samples.jsonlines").write_text(
            "".join(json.dumps(s) + "\n" for s in samples),
            encoding="utf-8",
        )

        dataset = DualGraphDataset(
            root=str(base / "cache"),
            dataset_dir=dataset_dir,
            num_workers=1,
        )

        assert len(dataset) == 2
        first = dataset[0]
        assert first.edge_index.shape == (2, 1)
        assert first.edge_labels.tolist() == [0]
        assert torch.allclose(first.y[0, [0, 2, 3]], torch.tensor([0.5, 0.0, 1.0]))
        assert math.isnan(first.y[0, 1].item())
        assert first.asm_node_type.numel() == 1

        second = dataset[1]
        assert second.asm_node_type.tolist() == [0]
        assert second.asm_instruction_encoding.shape == (1, 256)


def test_candidate_manifest_without_targets_loads_as_unlabeled() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        dataset_dir = base / "candidate_dataset"
        _write_json(dataset_dir / "rtl_graphs" / "ALU.json", _minimal_rtl_graph())
        _write_json(dataset_dir / "asm_graphs" / "candidate.json", _minimal_asm_graph())
        _write_json(
            dataset_dir / "manifest.json",
            {
                "schema_version": "dataset.v1",
                "mode": "candidate/inference",
                "dataset_name": "candidate-unit",
                "samples": "samples.jsonlines",
                "rtl_graphs": "rtl_graphs",
                "asm_graphs": "asm_graphs",
                "total": 1,
                "errors": 0,
            },
        )
        sample = _sample_entry("candidate::ALU", "asm_graphs/candidate.json")
        sample.pop("targets")
        (dataset_dir / "samples.jsonlines").write_text(
            json.dumps(sample) + "\n",
            encoding="utf-8",
        )

        dataset = DualGraphDataset(
            root=str(base / "cache"),
            dataset_dir=dataset_dir,
            num_workers=1,
        )

        assert len(dataset) == 1
        candidate = dataset[0]
        assert candidate.edge_labels.tolist() == [-1]
        assert torch.isnan(candidate.y).all()


def test_manifest_dataset_merges_multiple_dataset_dirs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        dataset_dirs = []
        for idx in range(2):
            dataset_dir = base / f"dataset_{idx}"
            dataset_dirs.append(dataset_dir)
            _write_json(dataset_dir / "rtl_graphs" / "ALU.json", _minimal_rtl_graph())
            _write_json(
                dataset_dir / "asm_graphs" / f"test_{idx}.json",
                _minimal_asm_graph(),
            )
            _write_json(
                dataset_dir / "manifest.json",
                {
                    "schema_version": "dataset.v1",
                    "dataset_name": f"unit_{idx}",
                    "samples": "samples.jsonlines",
                    "rtl_graphs": "rtl_graphs",
                    "asm_graphs": "asm_graphs",
                    "total": 1,
                    "errors": 0,
                },
            )
            sample = _sample_entry(
                f"test_{idx}::ALU",
                f"asm_graphs/test_{idx}.json",
            )
            sample["targets"]["graph_coverage"]["values"]["branch"] = 50.0 + idx
            (dataset_dir / "samples.jsonlines").write_text(
                json.dumps(sample) + "\n",
                encoding="utf-8",
            )

        dataset = DualGraphDataset(
            root=str(base / "cache"),
            dataset_dir=dataset_dirs,
            num_workers=1,
        )

        assert len(dataset) == 2
        assert torch.isclose(dataset[0].y[0, 0], torch.tensor(0.50))
        assert torch.isclose(dataset[1].y[0, 0], torch.tensor(0.51))


def test_vector_contrastive_manifest_builds_pair_train_datamodule() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        dataset_dir = base / "dataset"
        _write_json(dataset_dir / "rtl_graphs" / "ALU.json", _minimal_rtl_graph())
        _write_json(dataset_dir / "asm_graphs" / "test_a.json", _minimal_asm_graph())
        _write_json(
            dataset_dir / "manifest.json",
            {
                "schema_version": "dataset.v1",
                "dataset_name": "unit",
                "samples": "samples.jsonlines",
                "rtl_graphs": "rtl_graphs",
                "asm_graphs": "asm_graphs",
                "total": 3,
                "errors": 0,
            },
        )
        samples = [
            _sample_entry_with_vectors(
                "test_a::ALU",
                "asm_graphs/test_a.json",
                _coverage_vectors(line_ids=[0], branch_ids=[0]),
            ),
            _sample_entry_with_vectors(
                "test_b::ALU",
                None,
                _coverage_vectors(line_ids=[0], branch_ids=[1]),
            ),
            _sample_entry_with_vectors(
                "test_c::ALU",
                None,
                _coverage_vectors(line_ids=[0], branch_ids=[1]),
            ),
        ]
        (dataset_dir / "samples.jsonlines").write_text(
            "".join(json.dumps(s) + "\n" for s in samples),
            encoding="utf-8",
        )

        dm, has_validation = _build_training_datamodule(
            data_root=str(base / "cache"),
            dataset_dir=str(dataset_dir),
            text_encoder_config=None,
            trainer_config=TrainerConfig(
                joint_contrastive=True,
                batch_size=2,
                contrastive_batch_size=2,
                num_workers=0,
            ),
        )
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))

        assert has_validation is True
        assert batch.batch_a.edge_labels.numel() == 1
        assert batch.batch_b.edge_labels.numel() == 1
        assert torch.allclose(batch.similarity, torch.tensor([0.5]))


def test_joint_contrastive_pair_training_excludes_auto_val_indices() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        dataset_dir = base / "dataset"
        _write_json(dataset_dir / "rtl_graphs" / "ALU.json", _minimal_rtl_graph())
        _write_json(
            dataset_dir / "manifest.json",
            {
                "schema_version": "dataset.v1",
                "dataset_name": "unit",
                "samples": "samples.jsonlines",
                "rtl_graphs": "rtl_graphs",
                "asm_graphs": "asm_graphs",
                "total": 20,
                "errors": 0,
            },
        )
        samples = [
            _sample_entry_with_vectors(
                f"test_{idx}::ALU",
                None,
                _coverage_vectors(line_ids=[idx % 2], branch_ids=[idx % 2]),
            )
            for idx in range(20)
        ]
        (dataset_dir / "samples.jsonlines").write_text(
            "".join(json.dumps(s) + "\n" for s in samples),
            encoding="utf-8",
        )

        dm, _ = _build_training_datamodule(
            data_root=str(base / "cache"),
            dataset_dir=str(dataset_dir),
            text_encoder_config=None,
            trainer_config=TrainerConfig(
                joint_contrastive=True,
                batch_size=2,
                contrastive_batch_size=2,
                num_workers=0,
            ),
        )
        dm.setup("fit")

        train_indices = set(dm._base.train_indices)
        val_indices = set(dm._base.val_indices)
        pair_indices = {
            sample_idx
            for pair in dm._pair_dm._dataset._pairs
            for sample_idx in pair[:2]
        }
        val_pair_indices = {
            sample_idx
            for pair in dm._pair_val_dm._dataset._pairs
            for sample_idx in pair[:2]
        }

        assert train_indices
        assert val_indices
        assert pair_indices
        assert pair_indices <= train_indices
        assert pair_indices.isdisjoint(val_indices)
        assert val_pair_indices
        assert val_pair_indices <= val_indices
        assert val_pair_indices.isdisjoint(train_indices)

        val_loaders = dm.val_dataloader()
        assert isinstance(val_loaders, list)
        assert len(val_loaders) == 2
        pair_batch = next(iter(val_loaders[1]))
        assert pair_batch.jaccard.shape[1] == 5
        assert pair_batch.density_a.shape == pair_batch.jaccard.shape

        module = DualGraphLightningModule(
            model_config=ModelConfig(
                hidden_dim=16,
                num_gnn_layers=1,
                use_hyperrectangle=True,
            ),
            joint_contrastive=True,
        )
        trainer = L.Trainer(
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            limit_val_batches=1,
        )
        results = trainer.validate(module, datamodule=dm, verbose=False)

        assert len(results) == 2
        assert "val/total_loss" in results[0]
        assert "val/pair_iou_loss" in results[1]


if __name__ == "__main__":
    test_manifest_dataset_loads_sparse_targets_without_legacy_index()
    test_manifest_dataset_merges_multiple_dataset_dirs()
    test_vector_contrastive_manifest_builds_pair_train_datamodule()
    test_joint_contrastive_pair_training_excludes_auto_val_indices()
