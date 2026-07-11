import json

import pytest

from datasets.manifest import read_manifest_samples


def test_manifest_total_must_match_sample_records(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "dataset.v1",
                "samples": "samples.jsonlines",
                "total": 1,
            }
        ),
        encoding="utf-8",
    )
    records = [
        {
            "schema_version": "sample.v1",
            "sample_id": sample_id,
            "rtl_graph": "rtl.json",
            "asm_graph": None,
        }
        for sample_id in ("a", "b")
    ]
    (tmp_path / "samples.jsonlines").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest total.*sample records"):
        read_manifest_samples(tmp_path)
