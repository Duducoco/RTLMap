import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DATASETS = (
    "archgen_single",
    "picorv32",
    "riscv_simple_multicycle",
)


def _install_queue_fixture(tmp_path: Path) -> tuple[Path, Path]:
    queue_script = tmp_path / "run_contrastive_queue.sh"
    queue_script.write_text(
        (PROJECT_ROOT / "run_contrastive_queue.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    training_script = tmp_path / "run_contrastive.sh"
    training_script.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$1" >> "$QUEUE_CAPTURE"\n',
        encoding="utf-8",
    )
    training_script.chmod(0o755)
    capture_path = tmp_path / "datasets.txt"
    return queue_script, capture_path


def _start_queue(
    queue_script: Path,
    capture_path: Path,
    *args: str,
    extra_env: dict[str, str] | None = None,
) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["QUEUE_CAPTURE"] = str(capture_path)
    env.update(extra_env or {})
    return subprocess.Popen(
        ["bash", str(queue_script), *args],
        cwd=queue_script.parent,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _assert_waits_then_runs(
    queue: subprocess.Popen[str],
    blocker: subprocess.Popen[bytes],
    capture_path: Path,
) -> None:
    try:
        assert queue.stdout is not None
        first_line = queue.stdout.readline()
        assert "等待" in first_line
        assert not capture_path.exists()

        blocker.terminate()
        blocker.wait(timeout=5)
        stdout, stderr = queue.communicate(timeout=5)

        assert queue.returncode == 0, first_line + stdout + stderr
        assert capture_path.read_text(encoding="utf-8").splitlines() == list(
            EXPECTED_DATASETS
        )
    finally:
        if blocker.poll() is None:
            blocker.terminate()
            blocker.wait(timeout=5)
        if queue.poll() is None:
            queue.terminate()
            queue.wait(timeout=5)


def test_queue_waits_for_pid_then_runs_datasets_in_order(tmp_path: Path) -> None:
    queue_script, capture_path = _install_queue_fixture(tmp_path)

    blocker = subprocess.Popen(
        ["bash", "-c", "while :; do sleep 1; done"],
        cwd=tmp_path,
    )
    queue = _start_queue(
        queue_script,
        capture_path,
        "--wait-pid",
        str(blocker.pid),
        "--poll-interval",
        "0.01",
    )

    _assert_waits_then_runs(queue, blocker, capture_path)


def test_queue_auto_detects_current_contrastive_script(tmp_path: Path) -> None:
    queue_script, capture_path = _install_queue_fixture(tmp_path)

    current_dir = tmp_path / "current"
    current_dir.mkdir()
    current_training = current_dir / "run_contrastive.sh"
    current_training.write_text(
        "#!/usr/bin/env bash\nwhile :; do sleep 1; done\n",
        encoding="utf-8",
    )
    current_training.chmod(0o755)
    blocker = subprocess.Popen(["bash", str(current_training)], cwd=tmp_path)
    unrelated_dir = tmp_path / "unrelated"
    unrelated_dir.mkdir()
    unrelated = subprocess.Popen(
        ["bash", str(current_training)],
        cwd=unrelated_dir,
    )
    fake_pgrep = tmp_path / "pgrep"
    fake_pgrep.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$AUTO_DETECT_PIDS"\n',
        encoding="utf-8",
    )
    fake_pgrep.chmod(0o755)

    queue = _start_queue(
        queue_script,
        capture_path,
        "--poll-interval",
        "0.01",
        extra_env={
            "AUTO_DETECT_PIDS": f"{blocker.pid}\n{unrelated.pid}",
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
        },
    )

    try:
        _assert_waits_then_runs(queue, blocker, capture_path)
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_queue_help_describes_monitoring_and_dataset_order() -> None:
    result = subprocess.run(
        ["bash", str(PROJECT_ROOT / "run_contrastive_queue.sh"), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--wait-pid PID" in result.stdout
    assert "--poll-interval SECONDS" in result.stdout
    assert "自动检测" in result.stdout
    assert " -> ".join(EXPECTED_DATASETS) in result.stdout


def test_queue_rejects_invalid_poll_interval() -> None:
    result = subprocess.run(
        [
            "bash",
            str(PROJECT_ROOT / "run_contrastive_queue.sh"),
            "--wait-pid",
            str(os.getpid()),
            "--poll-interval",
            "invalid",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--poll-interval 必须是正数" in result.stderr
