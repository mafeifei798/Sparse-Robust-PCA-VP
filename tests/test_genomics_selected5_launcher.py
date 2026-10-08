from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools import run_genomics_selected5 as launcher


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("module_entry", [True, False])
def test_launcher_help_needs_no_external_pythonpath_or_cohort(tmp_path, module_entry):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    command = (
        [sys.executable, "-m", "tools.run_genomics_selected5"]
        if module_entry else [sys.executable, str(ROOT / "tools/run_genomics_selected5.py")]
    )
    result = subprocess.run(
        [*command, "--help"], cwd=ROOT if module_entry else tmp_path,
        env=environment, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--output" in result.stdout and "--protocol" in result.stdout
    assert "Traceback" not in result.stderr


def test_launcher_preserves_arguments_environment_and_exit_code(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "existing-path")
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 23)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    assert launcher.main(["--protocol", "custom protocol.md", "--output", "new-output"]) == 23
    command, kwargs = calls[0]
    assert command[2:] == ["--protocol", "custom protocol.md", "--output", "new-output"]
    assert kwargs["env"]["OMP_NUM_THREADS"] == "7"
    assert kwargs["env"]["PYTHONPATH"] == str(ROOT / "src") + os.pathsep + "existing-path"


def test_launcher_does_not_rewrite_source_bound_runner():
    runner = ROOT / "experiments/genomics_randomized_vp_selected5.py"
    assert sha256(runner.read_bytes()).hexdigest() == "f3c605ec8e343ad3b0408c72b54c41a49537df8a207a22dfa73b671bafb452be"
