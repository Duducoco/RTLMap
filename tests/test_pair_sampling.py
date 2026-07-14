"""Behavioral tests for relative-stratified Geometry Pair sampling."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import datasets.pair_datamodule as pair_module
from datasets.datamodule import DualGraphDataModule
from datasets.manifest import read_manifest_samples
from datasets.pair_datamodule import ContrastivePairDataset
from trainer.datamodule_factory import JointTrainDataModule
from trainer.config import TrainerConfig
from trainer.lightning_trainer import LightningTrainer


def _vectors(*, element_count: int, covered: set[int]) -> dict:
    return {
        "line": {
            "element_count": element_count,
            "items": [{"id": item_id, "label": 1} for item_id in sorted(covered)],
        },
        "condition": {"element_count": 0, "items": []},
        "toggle": {"element_count": 0, "items": []},
        "fsm": {"element_count": 0, "items": []},
        "branch": {"element_count": 0, "items": []},
    }


def _write_dataset(root: Path, samples: list[dict]) -> None:
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "dataset.v1",
                "samples": "samples.jsonlines",
                "total": len(samples),
            }
        ),
        encoding="utf-8",
    )
    (root / "samples.jsonlines").write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples),
        encoding="utf-8",
    )


def _samples() -> list[dict]:
    samples: list[dict] = []
    universe = set(range(32))
    for module_name in ("HIGH", "LOW"):
        for index in range(20):
            if module_name == "HIGH":
                covered = universe - {index % 16, (index + 1) % 16}
            else:
                covered = {
                    (index * 3) % 32,
                    (index * 3 + 1) % 32,
                    (index * 7 + 5) % 32,
                }
            sample_id = f"test_{index:02d}::{module_name}"
            samples.append(
                {
                    "schema_version": "sample.v1",
                    "sample_id": sample_id,
                    "test_id": f"test_{index:02d}",
                    "module_name": module_name,
                    "rtl_graph": f"rtl_graphs/{module_name}.json",
                    "asm_graph": None,
                    "targets": {
                        "coverage_vectors": _vectors(
                            element_count=32,
                            covered=covered,
                        )
                    },
                }
            )
    return samples


class _FakeGraphDataset:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    def __len__(self) -> int:
        return 10_000

    def __getitem__(self, index: int) -> int:
        return index


@pytest.fixture
def pair_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, _samples())
    monkeypatch.setattr(pair_module, "DualGraphDataset", _FakeGraphDataset)
    return ContrastivePairDataset(
        dataset_dir=dataset_dir,
        candidate_pool_size=128,
        relative_low_quota=2,
        relative_mid_quota=1,
        relative_high_quota=1,
        sampling_seed=17,
    )


def _unordered_pairs(dataset: ContrastivePairDataset) -> set[frozenset[int]]:
    return {frozenset((record.idx_a, record.idx_b)) for record in dataset.pair_records}


def test_pair_sampling_is_unique_relative_and_epoch_deterministic(
    pair_dataset: ContrastivePairDataset,
) -> None:
    records_epoch_zero = tuple(pair_dataset.pair_records)
    unordered_epoch_zero = _unordered_pairs(pair_dataset)

    assert len(unordered_epoch_zero) == len(records_epoch_zero)
    assert {record.stratum for record in records_epoch_zero} == {
        "relative-low",
        "relative-mid",
        "relative-high",
    }
    assert {
        pair_dataset.samples[record.idx_a]["module_name"]
        for record in records_epoch_zero
    } == {"HIGH", "LOW"}

    pair_dataset.resample(1)
    records_epoch_one = tuple(pair_dataset.pair_records)
    assert records_epoch_one != records_epoch_zero

    pair_dataset.resample(0)
    assert tuple(pair_dataset.pair_records) == records_epoch_zero
    assert _unordered_pairs(pair_dataset) == unordered_epoch_zero


def test_pair_sampling_is_independent_of_manifest_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    samples = _samples()
    _write_dataset(dataset_dir, samples)
    monkeypatch.setattr(pair_module, "DualGraphDataset", _FakeGraphDataset)

    first = ContrastivePairDataset(
        dataset_dir=dataset_dir,
        candidate_pool_size=128,
        relative_low_quota=2,
        relative_mid_quota=1,
        relative_high_quota=1,
        sampling_seed=23,
    )
    first_fingerprint = first.pair_fingerprint

    (dataset_dir / "samples.jsonlines").write_text(
        "".join(json.dumps(sample) + "\n" for sample in reversed(samples)),
        encoding="utf-8",
    )
    reordered = ContrastivePairDataset(
        dataset_dir=dataset_dir,
        candidate_pool_size=128,
        relative_low_quota=2,
        relative_mid_quota=1,
        relative_high_quota=1,
        sampling_seed=23,
    )

    assert reordered.pair_fingerprint == first_fingerprint


def test_grouped_split_keeps_test_stimuli_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    samples = _samples()
    _write_dataset(dataset_dir, samples)
    module = DualGraphDataModule(
        root=str(tmp_path / "cache"),
        dataset_dir=dataset_dir,
        num_workers=0,
    )
    fake_dataset = list(range(len(samples)))
    monkeypatch.setattr(module, "_make_dataset", lambda split: fake_dataset)

    module.setup("fit")

    manifest_samples = read_manifest_samples(dataset_dir)
    train_stimuli = {
        (sample["_dataset_dir"], sample["test_id"])
        for index in module.train_indices
        if (sample := manifest_samples[index])
    }
    val_stimuli = {
        (sample["_dataset_dir"], sample["test_id"])
        for index in module.val_indices
        if (sample := manifest_samples[index])
    }
    train_modules = {
        manifest_samples[index]["module_name"] for index in module.train_indices
    }
    val_modules = {
        manifest_samples[index]["module_name"] for index in module.val_indices
    }

    assert val_stimuli
    assert train_stimuli.isdisjoint(val_stimuli)
    assert val_modules <= train_modules


def test_endpoint_weights_equalize_full_epoch_sample_contribution(
    pair_dataset: ContrastivePairDataset,
) -> None:
    contribution: dict[int, float] = defaultdict(float)
    observed_degree: dict[int, int] = defaultdict(int)
    for record in pair_dataset.pair_records:
        contribution[record.idx_a] += record.endpoint_weight_a
        contribution[record.idx_b] += record.endpoint_weight_b
        observed_degree[record.idx_a] += 1
        observed_degree[record.idx_b] += 1

    assert contribution
    assert max(contribution.values()) == pytest.approx(min(contribution.values()))
    for record in pair_dataset.pair_records:
        assert record.degree_a == observed_degree[record.idx_a]
        assert record.degree_b == observed_degree[record.idx_b]


def test_joint_train_loader_resamples_for_current_epoch() -> None:
    class EpochAwarePairDataModule:
        def __init__(self):
            self.epoch = -1

        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch

        def train_dataloader(self) -> int:
            return self.epoch

    pair_dm = EpochAwarePairDataModule()
    joint = JointTrainDataModule(
        base=SimpleNamespace(),
        pair_dm=pair_dm,
        pair_val_dm=SimpleNamespace(),
    )
    joint._trainer = SimpleNamespace(current_epoch=0)
    assert joint.train_dataloader() == 0

    joint._trainer.current_epoch = 1
    assert joint.train_dataloader() == 1


def test_sampling_diagnostics_report_bounded_pool_and_quota_fulfillment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, _samples())
    monkeypatch.setattr(pair_module, "DualGraphDataset", _FakeGraphDataset)
    dataset = ContrastivePairDataset(
        dataset_dir=dataset_dir,
        candidate_pool_size=5,
        relative_low_quota=2,
        relative_mid_quota=1,
        relative_high_quota=1,
        sampling_seed=31,
    )

    diagnostics = dataset.sampling_diagnostics
    assert diagnostics["candidate_pool_max"] == 5
    assert diagnostics["eligible_anchors"] == 40
    assert diagnostics["unique_pairs"] == len(dataset)
    assert diagnostics["pair_fingerprint"] == dataset.pair_fingerprint
    assert diagnostics["fulfilled_relative_low"] > 0
    assert diagnostics["fulfilled_relative_mid"] > 0
    assert diagnostics["fulfilled_relative_high"] > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pair_candidate_pool_size": 0},
        {"pair_relative_low_quota": -1},
        {
            "pair_candidate_pool_size": 3,
            "pair_relative_low_quota": 2,
            "pair_relative_mid_quota": 1,
            "pair_relative_high_quota": 1,
        },
    ],
)
def test_pair_sampling_config_rejects_invalid_values(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        TrainerConfig(**kwargs)


def test_joint_trainer_reloads_dataloaders_each_epoch(tmp_path: Path) -> None:
    wrapper = LightningTrainer(
        TrainerConfig(
            joint_contrastive=True,
            accelerator="cpu",
            devices=1,
            checkpoint_dir=str(tmp_path),
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        ),
        logger_type="csv",
        has_validation=False,
    )

    assert wrapper.trainer.reload_dataloaders_every_n_epochs == 1
