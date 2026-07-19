import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAMES = (
    "archgen_single",
    "ibex",
    "picorv32",
    "riscv_simple_multicycle",
)


def _run_script(
    tmp_path: Path,
    checkpoint: str | None,
    *script_args: str,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    capture_path = tmp_path / "uv-args.txt"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$@" > "$CAPTURE_PATH"\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    dataset_root = tmp_path / "datasets"
    for dataset_name in DATASET_NAMES:
        dataset_dir = dataset_root / dataset_name
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "manifest.json").write_text("{}", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        BASH_COMPAT="4.3",
        CAPTURE_PATH=str(capture_path),
        DATA_ROOT="./stale_data_root",
        DATASET_NAME="stale_dataset",
        DATASET_ROOT=str(dataset_root),
        EXPERIMENT_NAME="stale_experiment",
        PATH=f"{tmp_path}:{env['PATH']}",
    )
    if checkpoint is None:
        env.pop("CKPT_PATH", None)
    else:
        env["CKPT_PATH"] = checkpoint

    result = subprocess.run(
        ["bash", "run_contrastive.sh", *script_args],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    args = capture_path.read_text(encoding="utf-8").splitlines() if capture_path.exists() else []
    return result, args


def _argument_value(args: list[str], option: str) -> str:
    return args[args.index(option) + 1]


def test_run_contrastive_without_checkpoint_is_nounset_safe(tmp_path: Path) -> None:
    result, args = _run_script(tmp_path, checkpoint=None)

    assert result.returncode == 0, result.stderr
    assert "--ckpt-path" not in args
    assert _argument_value(args, "--data-root") == "./data_contrastive_ibex"


def test_run_contrastive_passes_checkpoint_when_requested(tmp_path: Path) -> None:
    checkpoint = "/tmp/contrastive model.ckpt"
    result, args = _run_script(tmp_path, checkpoint=checkpoint)

    assert result.returncode == 0, result.stderr
    index = args.index("--ckpt-path")
    assert args[index + 1] == checkpoint


def test_run_contrastive_selects_dataset_and_derived_defaults(tmp_path: Path) -> None:
    result, args = _run_script(tmp_path, None, "--dataset", "picorv32")

    assert result.returncode == 0, result.stderr
    assert _argument_value(args, "--dataset-dir") == str(
        tmp_path / "datasets" / "picorv32"
    )
    assert _argument_value(args, "--data-root") == "./data_contrastive_picorv32"
    assert (
        _argument_value(args, "--experiment-name")
        == "picorv32-4coverage-split-rect-mlp-asm-readout"
    )
    assert _argument_value(args, "--graph-relative-loss-weight") == "0.1"
    assert _argument_value(args, "--graph-relative-loss-floor") == "0.1"


def test_run_contrastive_accepts_dataset_name_as_positional_argument(
    tmp_path: Path,
) -> None:
    result, args = _run_script(tmp_path, None, "riscv_simple_multicycle")

    assert result.returncode == 0, result.stderr
    assert _argument_value(args, "--dataset-dir") == str(
        tmp_path / "datasets" / "riscv_simple_multicycle"
    )
    assert (
        _argument_value(args, "--data-root")
        == "./data_contrastive_riscv_simple_multicycle"
    )


def test_run_contrastive_accepts_data_root_and_checkpoint_options(tmp_path: Path) -> None:
    data_root = str(tmp_path / "processed data")
    checkpoint = str(tmp_path / "checkpoints" / "last model.ckpt")

    result, args = _run_script(
        tmp_path,
        None,
        "--dataset",
        "ibex",
        "--data-root",
        data_root,
        "--ckpt-path",
        checkpoint,
    )

    assert result.returncode == 0, result.stderr
    assert _argument_value(args, "--data-root") == data_root
    assert _argument_value(args, "--ckpt-path") == checkpoint


def test_run_contrastive_help_describes_dataset_selection(tmp_path: Path) -> None:
    result, args = _run_script(tmp_path, None, "--help")

    assert result.returncode == 0, result.stderr
    assert "--dataset NAME" in result.stdout
    assert "--data-root PATH" in result.stdout
    assert "--ckpt-path PATH" in result.stdout
    for dataset_name in DATASET_NAMES:
        assert f"bash run_contrastive.sh {dataset_name}" in result.stdout
    assert args == []
