import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_run_train_passes_selected_model_architecture(tmp_path: Path) -> None:
    capture_path = tmp_path / "uv-args.txt"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$CAPTURE_PATH"\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    dataset_root = tmp_path / "datasets"
    dataset_dir = dataset_root / "ibex"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "manifest.json").write_text("{}", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        CAPTURE_PATH=str(capture_path),
        DATASET_ROOT=str(dataset_root),
        MODEL_ARCHITECTURE="pooled_add",
        PATH=f"{tmp_path}:{env['PATH']}",
    )

    result = subprocess.run(
        ["bash", "run_train.sh", "--dataset", "ibex"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    args = capture_path.read_text(encoding="utf-8").splitlines()
    architecture_index = args.index("--model-architecture")
    assert args[architecture_index + 1] == "pooled_add"
