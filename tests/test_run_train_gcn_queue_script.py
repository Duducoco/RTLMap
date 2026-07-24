import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DATASETS = (
    "archgen_single",
    "ibex",
    "picorv32",
    "riscv_simple_multicycle",
)


def _install_queue_fixture(tmp_path: Path) -> tuple[Path, Path]:
    queue_script = tmp_path / "run_train_gcn_queue.sh"
    queue_script.write_text(
        (PROJECT_ROOT / "run_train_gcn_queue.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    training_script = tmp_path / "run_train.sh"
    training_script.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s|%s|%s|%s\\n\' "$2" "$MODEL_ARCHITECTURE" '
        '"$EXPERIMENT_NAME" "$CHECKPOINT_DIR" >> "$QUEUE_CAPTURE"\n',
        encoding="utf-8",
    )
    training_script.chmod(0o755)
    return queue_script, tmp_path / "queue.txt"


def test_gcn_queue_runs_every_dataset_with_isolated_experiments(tmp_path: Path) -> None:
    queue_script, capture_path = _install_queue_fixture(tmp_path)
    env = os.environ.copy()
    env["QUEUE_CAPTURE"] = str(capture_path)

    result = subprocess.run(
        ["bash", str(queue_script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    rows = [line.split("|") for line in capture_path.read_text().splitlines()]
    assert [row[0] for row in rows] == list(EXPECTED_DATASETS)
    assert {row[1] for row in rows} == {"rtl_gcn"}
    assert [row[2] for row in rows] == [
        f"{dataset}-4coverage-rtl-gcn" for dataset in EXPECTED_DATASETS
    ]
    assert {row[3] for row in rows} == {"checkpoints/rtl_gcn"}


def test_gcn_queue_help_reports_architecture_and_order() -> None:
    result = subprocess.run(
        ["bash", str(PROJECT_ROOT / "run_train_gcn_queue.sh"), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "rtl_gcn" in result.stdout
    assert " -> ".join(EXPECTED_DATASETS) in result.stdout
