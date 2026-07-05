#!/usr/bin/env python3
"""coverage-report-extractor manifest 读取工具。"""

import json
from pathlib import Path
from typing import Optional, Sequence, Union

DatasetDirInput = Union[str, Path, Sequence[Union[str, Path]]]


def normalize_dataset_dirs(dataset_dir: Optional[DatasetDirInput]) -> list[Path]:
    if dataset_dir is None:
        return []
    if isinstance(dataset_dir, (str, Path)):
        return [Path(dataset_dir)]
    return [Path(p) for p in dataset_dir]


def read_manifest_samples(dataset_dir: Path) -> list[dict]:
    """读取 ``dataset.v1`` manifest/samples。"""
    manifest_file = dataset_dir / "manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"manifest.json 不存在: {manifest_file}")

    with open(manifest_file, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("schema_version") != "dataset.v1":
        raise ValueError(
            f"不支持的数据集 schema: {manifest.get('schema_version')!r}, "
            "期望 dataset.v1"
        )

    samples_file = dataset_dir / manifest.get("samples", "samples.jsonlines")
    if not samples_file.exists():
        raise FileNotFoundError(f"samples.jsonlines 不存在: {samples_file}")

    results: list[dict] = []
    with open(samples_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("schema_version") != "sample.v1":
                raise ValueError(
                    f"不支持的样本 schema: {entry.get('schema_version')!r}"
                )
            sample = dict(entry)
            sample["_rtl_path"] = str(dataset_dir / entry["rtl_graph"])
            sample["_asm_path"] = (
                str(dataset_dir / entry["asm_graph"])
                if entry.get("asm_graph")
                else None
            )
            results.append(sample)
    return results


def read_manifest_samples_from_dirs(dataset_dirs: list[Path]) -> list[dict]:
    samples: list[dict] = []
    for dataset_dir in dataset_dirs:
        samples.extend(read_manifest_samples(dataset_dir))
    return samples


def read_manifest(dataset_dir: Path) -> dict:
    manifest_file = dataset_dir / "manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"manifest.json 不存在: {manifest_file}")
    with open(manifest_file, encoding="utf-8") as f:
        return json.load(f)
