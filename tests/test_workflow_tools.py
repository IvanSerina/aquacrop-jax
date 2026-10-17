"""Exercise recovery orchestration with isolated files and no local model calls."""
import argparse
from contextlib import contextmanager
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.error

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sandbox(tmp_path):
    root = tmp_path / "repo"
    (root / "tools").mkdir(parents=True)
    for name in ("cpu-env.sh", "workflow.py", "test-and-capture", "failure-gate", "delegate-local"):
        shutil.copy2(ROOT / "tools" / name, root / "tools" / name)
    # This fixture tests dispatch and guard behavior, not JAX numerical kernels.
    (root / "src").mkdir()
    (root / "src/jax.py").write_text(
        "import os\nfrom types import SimpleNamespace\n__version__ = 'test'\n"
        "config = SimpleNamespace(jax_enable_x64=True)\n"
        "def default_backend(): return os.environ.get('FAKE_JAX_BACKEND', 'cpu')\n"
        "def devices(): return [SimpleNamespace(platform=default_backend())]\n")
    (root / "src/jaxlib.py").write_text("__version__ = 'test'\n")
    return root


def invoke(root, *args, updates=None):
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "none",
                "JAX_PLATFORMS": "cpu", "JAX_PLATFORM_NAME": "cpu",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "AQUACROP_CPU_THREADS": "1",
                "AQUACROP_PYTHON": sys.executable})
    env.update(updates or {})
    return subprocess.run([str(root / "tools/test-and-capture"), *args],
                          cwd=root, env=env, text=True, capture_output=True, timeout=30)


def test_capture_records_cpu_environment_and_output(sandbox):
    result = invoke(sandbox, "--", sys.executable, "-c",
                    "import os; assert os.environ['JAX_PLATFORMS'] == 'cpu'; "
                    "assert os.environ['CUDA_VISIBLE_DEVICES'] == ''; print('captured')")
    assert result.returncode == 0, result.stderr
    path, = (sandbox / ".sequence3/dossiers").iterdir()
    assert (path / "targeted.log").read_text().strip() == "captured"
    assert json.loads((path / "targeted.status.json").read_text())["exit_code"] == 0
    environment = json.loads((path / "environment.json").read_text())
    assert environment["backend"] == "cpu"
    assert environment["dtype"] == "float64"
    assert environment["environment"]["OMP_NUM_THREADS"] == "1"


@pytest.mark.parametrize("gate_code", [0, 13])
def test_failed_command_waits_for_gate_and_preserves_original_exit(sandbox, gate_code):
    (sandbox / "tools/failure-gate").write_text(
        '#!/usr/bin/env bash\nset -eu\n'
        '[[ "$1" == "--dossier" ]]\n'
        'printf "complete\\n" > "$2/gate.finished"\n'
        f'exit {gate_code}\n')
    result = invoke(sandbox, "--", sys.executable, "-c", "raise SystemExit(7)")
    assert result.returncode == 7, result.stderr
    path, = (sandbox / ".sequence3/dossiers").iterdir()
    assert (path / "gate.finished").read_text().strip() == "complete"
    assert json.loads((path / "failure-gate.status.json").read_text())["exit_code"] == gate_code


@pytest.mark.parametrize("updates", [{"CUDA_VISIBLE_DEVICES": "0"}, {"FAKE_JAX_BACKEND": "gpu"}])
def test_incompatible_backend_stops_before_requested_command(sandbox, updates):
    marker = sandbox / "must-not-exist"
    result = invoke(sandbox, "--", sys.executable, "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).touch()", updates=updates)
    assert result.returncode == 78
    assert not marker.exists()
    assert not (sandbox / ".sequence3").exists()


def test_leading_pytest_options_are_forwarded(sandbox):
    (sandbox / "src/pytest.py").write_text("import sys; print(repr(sys.argv[1:]))\n")
    result = invoke(sandbox, "-q", "tests/test_example.py")
    assert result.returncode == 0, result.stderr
    path, = (sandbox / ".sequence3/dossiers").iterdir()
    assert (path / "targeted.log").read_text().strip() == "['-q', 'tests/test_example.py']"


@pytest.fixture
def workflow(sandbox, monkeypatch):
    spec = importlib.util.spec_from_file_location("isolated_workflow", sandbox / "tools/workflow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("AQUACROP_CPU_THREADS", "1")
    monkeypatch.setenv("AQUACROP_OLLAMA_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("AQUACROP_LOCAL_MODEL", "test:local")
    return module


def test_failure_gate_continues_collecting_after_pytest_failure(workflow, monkeypatch):
    reference = workflow.REFERENCE / "testcase/LIST"
    reference.mkdir(parents=True)
    (reference / "ListProjects.txt").write_text("Ottawa.PRM\n")
    path = workflow.dossier("failure")
    monkeypatch.setattr(workflow, "git_snapshot", lambda target: None)
    monkeypatch.setattr(workflow, "benchmark_jax", lambda target: {"scope": "mock"})
    calls = []

    def failed_capture(command, log, **kwargs):
        calls.append(log.name)
        log.write_text("simulated failure\n")
        return {"command": command, "exit_code": 1, "seconds": 0.}

    monkeypatch.setattr(workflow, "capture", failed_capture)
    assert workflow.failure_gate(path, {"backend": "cpu"}) == 0
    assert calls == ["pytest-full.log", "stage-climate.log", "stage-water.log",
                     "stage-water-evap.log", "stage-water-crop.log"]
    assert (path / "gate-summary.json").exists()
    assert (workflow.ROOT / ".sequence3/NEXT_RUN.md").exists()


def test_local_unavailable_leaves_diagnosis_without_cloud_fallback(workflow, monkeypatch):
    class Offline:
        def open(self, *args, **kwargs):
            raise urllib.error.URLError("local service unavailable")

    monkeypatch.setattr(workflow.urllib.request, "build_opener", lambda *args: Offline())
    result = workflow.delegate(argparse.Namespace(task="Map one routine", file=[]), {"backend": "cpu"})
    assert result == 69
    path, = (workflow.ROOT / ".sequence3/dossiers").iterdir()
    status = json.loads((path / "status.json").read_text())
    assert "local service unavailable" in status["reason"]
    assert (path / "task.txt").read_text().strip() == "Map one routine"
    assert "AQUACROP_LOCAL_MODEL" in (workflow.ROOT / ".sequence3/NEXT_RUN.md").read_text()


def test_local_generation_is_cpu_only_bounded_and_read_only(workflow, monkeypatch):
    requests = []

    class Local:
        @contextmanager
        def open(self, request, **kwargs):
            if isinstance(request, str):
                yield io.StringIO(json.dumps({"models": [{"name": "test:local"}]}))
            else:
                requests.append(json.loads(request.data))
                yield io.StringIO(json.dumps({"done": True, "response": "Routine mapped"}))

    monkeypatch.setattr(workflow.urllib.request, "build_opener", lambda *args: Local())
    source = workflow.ROOT / "src/example.py"
    source.write_text("original = True\nexcluded = True\n")
    result = workflow.delegate(argparse.Namespace(task="Map one routine", file=["src/example.py:1:1"]), {})
    assert result == 0
    assert source.read_text() == "original = True\nexcluded = True\n"
    assert requests[0]["options"]["num_gpu"] == 0
    assert requests[0]["options"]["num_thread"] == 1
    assert requests[0]["keep_alive"] == 0
    assert "original = True" in requests[0]["prompt"]
    assert "excluded = True" not in requests[0]["prompt"]


def test_external_model_endpoint_is_refused_before_network_access(workflow, monkeypatch):
    monkeypatch.setenv("AQUACROP_OLLAMA_URL", "https://example.com")
    monkeypatch.setattr(workflow.urllib.request, "build_opener",
                        lambda *args: pytest.fail("external network must never be used"))
    with pytest.raises(RuntimeError, match="loopback"):
        workflow.delegate(argparse.Namespace(task="Map one routine", file=[]), {})
