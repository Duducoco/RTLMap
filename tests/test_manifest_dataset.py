#!/usr/bin/env python3
"""Tests for the coverage-report-extractor manifest dataset format."""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import DualGraphDataset
from datasets.triple_datamodule import ContrastiveTripleDataset
from trainer.config import TrainerConfig
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


def _contrastive_entry() -> dict:
    target = _sample_entry("test_a::ALU", "asm_graphs/test_a.json")["targets"]
    target_b = json.loads(json.dumps(target))
    target_b["edge_coverage"]["labels"] = [{"edge_id": 1, "label": 1}]
    target_b["graph_coverage"]["values"]["branch"] = 25.0
    target_merged = json.loads(json.dumps(target))
    target_merged["edge_coverage"]["labels"] = [{"edge_id": 1, "label": 1}]
    target_merged["graph_coverage"]["values"]["branch"] = 75.0
    return {
        "schema_version": "contrastive_sample.v1",
        "sample_id": "test_a::test_b::ALU",
        "module_name": "ALU",
        "rtl_graph": "rtl_graphs/ALU.json",
        "rtl_graph_hash": "sha256:test",
        "test_a_id": "test_a",
        "test_b_id": "test_b",
        "asm_a": "asm_graphs/test_a.json",
        "asm_b": None,
        "asm_a_hash": None,
        "asm_b_hash": None,
        "targets": {
            "a": target,
            "b": target_b,
            "merged": target_merged,
        },
    }


def test_contrastive_manifest_dataset_uses_shared_graph_and_targets() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        dataset_dir = base / "contrastive"
        _write_json(dataset_dir / "rtl_graphs" / "ALU.json", _minimal_rtl_graph())
        _write_json(dataset_dir / "asm_graphs" / "test_a.json", _minimal_asm_graph())
        _write_json(
            dataset_dir / "manifest.json",
            {
                "schema_version": "contrastive_dataset.v1",
                "dataset_name": "unit",
                "samples": "contrastive_samples.jsonlines",
                "rtl_graphs": "rtl_graphs",
                "asm_graphs": "asm_graphs",
                "total": 1,
                "errors": 0,
            },
        )
        (dataset_dir / "contrastive_samples.jsonlines").write_text(
            json.dumps(_contrastive_entry()) + "\n",
            encoding="utf-8",
        )

        dataset = ContrastiveTripleDataset(dataset_dir=dataset_dir)

        assert len(dataset) == 1
        data_a, data_b, data_merged, similarity = dataset[0]
        assert data_a.edge_labels.tolist() == [0]
        assert data_b.edge_labels.tolist() == [1]
        assert data_merged.edge_labels.tolist() == [1]
        assert data_b.asm_instruction_encoding.shape == (1, 256)
        assert 0.0 <= similarity <= 1.0


def test_contrastive_similarity_averages_all_coverage_targets() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        dataset_dir = base / "contrastive"
        _write_json(dataset_dir / "rtl_graphs" / "ALU.json", _minimal_rtl_graph())
        _write_json(dataset_dir / "asm_graphs" / "test_a.json", _minimal_asm_graph())
        _write_json(
            dataset_dir / "manifest.json",
            {
                "schema_version": "contrastive_dataset.v1",
                "dataset_name": "unit",
                "samples": "contrastive_samples.jsonlines",
                "rtl_graphs": "rtl_graphs",
                "asm_graphs": "asm_graphs",
                "total": 1,
                "errors": 0,
            },
        )

        entry = _contrastive_entry()
        entry["targets"]["a"]["graph_coverage"]["values"] = {
            "branch": 50.0,
            "line": 40.0,
            "fsm": 20.0,
            "toggle": 80.0,
            "condition": 30.0,
        }
        entry["targets"]["b"]["graph_coverage"]["values"] = {
            "branch": 25.0,
            "line": 20.0,
            "fsm": 10.0,
            "toggle": 60.0,
            "condition": 15.0,
        }
        entry["targets"]["merged"]["graph_coverage"]["values"] = {
            "branch": 75.0,
            "line": 50.0,
            "fsm": 25.0,
            "toggle": 100.0,
            "condition": 40.0,
        }
        (dataset_dir / "contrastive_samples.jsonlines").write_text(
            json.dumps(entry) + "\n",
            encoding="utf-8",
        )

        dataset = ContrastiveTripleDataset(dataset_dir=dataset_dir)

        _, _, _, similarity = dataset[0]
        expected = (0.0 + 0.2 + 0.2 + 0.4 + 0.125) / 5.0
        assert math.isclose(similarity, expected, rel_tol=1e-6)


def test_contrastive_manifest_builds_joint_train_datamodule_without_index() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        dataset_dir = base / "contrastive"
        _write_json(dataset_dir / "rtl_graphs" / "ALU.json", _minimal_rtl_graph())
        _write_json(dataset_dir / "asm_graphs" / "test_a.json", _minimal_asm_graph())
        _write_json(
            dataset_dir / "manifest.json",
            {
                "schema_version": "contrastive_dataset.v1",
                "dataset_name": "unit",
                "samples": "contrastive_samples.jsonlines",
                "rtl_graphs": "rtl_graphs",
                "asm_graphs": "asm_graphs",
                "total": 1,
                "errors": 0,
            },
        )
        (dataset_dir / "contrastive_samples.jsonlines").write_text(
            json.dumps(_contrastive_entry()) + "\n",
            encoding="utf-8",
        )

        dm, has_validation = _build_training_datamodule(
            data_root=str(base / "cache"),
            dataset_dir=str(dataset_dir),
            text_encoder_config=None,
            trainer_config=TrainerConfig(
                joint_contrastive=True,
                contrastive_triple_index="",
                batch_size=2,
                contrastive_batch_size=2,
                num_workers=0,
            ),
        )
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))

        assert has_validation is False
        assert dm.val_dataloader() == []
        assert dm.test_dataloader() == []
        assert batch.batch_a.edge_labels.numel() == 1
        assert batch.batch_b.edge_labels.numel() == 1
        assert batch.batch_merged.edge_labels.numel() == 1


if __name__ == "__main__":
    test_manifest_dataset_loads_sparse_targets_without_legacy_index()
    test_manifest_dataset_merges_multiple_dataset_dirs()
    test_contrastive_manifest_dataset_uses_shared_graph_and_targets()
    test_contrastive_similarity_averages_all_coverage_targets()
    test_contrastive_manifest_builds_joint_train_datamodule_without_index()
