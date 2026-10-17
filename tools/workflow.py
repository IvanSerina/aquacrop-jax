"""CPU-only recovery tools. No cloud calls and no automatic patch application."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / "aquacrop-fortran"
PYTHON = sys.executable
CPU_ENV = {"CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "none",
           "JAX_PLATFORMS": "cpu", "JAX_PLATFORM_NAME": "cpu",
           "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def preflight():
    for name, expected in CPU_ENV.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"CPU-only: configurazione incompatibile: {name}")
    import jax
    import jaxlib
    backend = jax.default_backend()
    devices = jax.devices()
    if backend != "cpu" or not devices or any(d.platform != "cpu" for d in devices):
        raise RuntimeError(f"CPU-only: backend rifiutato: {backend}; {devices}")
    if not jax.config.jax_enable_x64:
        raise RuntimeError("La parità richiede JAX float64")
    cpu = Path("/proc/cpuinfo")
    cpu_text = cpu.read_text() if cpu.exists() else ""
    model = next((line.split(":", 1)[1].strip() for line in cpu_text.splitlines()
                  if line.startswith("model name")), "unknown") if cpu.exists() else "unknown"
    cores = set()
    for block in cpu_text.split("\n\n"):
        fields = dict(line.split(":", 1) for line in block.splitlines() if ":" in line)
        fields = {key.strip(): value.strip() for key, value in fields.items()}
        if "physical id" in fields and "core id" in fields:
            cores.add((fields["physical id"], fields["core id"]))
    return {"python": sys.version, "executable": PYTHON, "jax": jax.__version__,
            "jaxlib": jaxlib.__version__, "backend": backend,
            "devices": [str(d) for d in devices], "dtype": "float64",
            "cpu_model": model, "physical_cores": len(cores) or None, "logical_cpus": os.cpu_count(),
            "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "environment": {key: os.environ.get(key) for key in
                            (*CPU_ENV, "AQUACROP_CPU_THREADS", "OMP_NUM_THREADS",
                             "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "XLA_FLAGS")}}


def dossier(label):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = ROOT / ".sequence3" / "dossiers" / f"{stamp}-{label}-{uuid4().hex[:8]}"
    path.mkdir(parents=True)
    return path


def capture(command, log, *, cwd=ROOT, timeout=900):
    start = time.perf_counter()
    print(f"{log.name}: avvio", flush=True)
    with log.open("w") as stream:
        try:
            result = subprocess.run(command, cwd=cwd, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=timeout)
            code = result.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            stream.write(f"\n{type(exc).__name__}: {exc}\n")
            code = 124 if isinstance(exc, subprocess.TimeoutExpired) else 127
    status = {"command": [str(arg) for arg in command], "cwd": str(cwd),
              "exit_code": code, "seconds": time.perf_counter() - start}
    save(log.with_suffix(".status.json"), status)
    print(f"{log.name}: exit={code}, {status['seconds']:.3f}s", flush=True)
    return status


def git_snapshot(path):
    for name, args in [("git-status.txt", ["status", "--short"]),
                       ("git-diff.patch", ["diff", "--binary", "HEAD"]),
                       ("git-last-commit.txt", ["log", "-1", "--format=fuller"]),
                       ("git-untracked.txt", ["ls-files", "--others", "--exclude-standard"])]:
        with (path / name).open("w") as stream:
            subprocess.run(["git", *args], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    # git diff omits untracked files; retain source copies without copying secrets.
    for base in (ROOT / "tools", ROOT / "src", ROOT / "tests"):
        if base.exists():
            shutil.copytree(base, path / "sources" / base.name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def next_run(path, reason, action):
    target = ROOT / ".sequence3" / "NEXT_RUN.md"
    # The previous diagnosis remains in its dossier instead of being discarded.
    if target.exists():
        shutil.copy2(target, path / "previous-NEXT_RUN.md")
    target.write_text(f"# Prossima sessione\n\nDossier: `{path.relative_to(ROOT)}`\n\n"
                      f"## Stato e causa\n\n{reason}\n\n"
                      "## Tentativi\n\nConsultare i log e gli exit code nel dossier. "
                      "Non ripetere lo stesso test senza nuove evidenze o modifiche.\n\n"
                      f"## Prossimo passo\n\n{action}\n\n"
                      "Tutte le esecuzioni numeriche devono rimanere CPU-only e float64. "
                      "La parità legacy completa non è dimostrata.\n")


def checksums(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.glob("*.OUT"))}


def benchmark_jax(path):
    """Time one explicit compiled GDD kernel; this is not a seasonal benchmark."""
    import jax
    import jax.numpy as jnp
    from aquacrop_jax.kernels import degrees_day
    climate = path / "climate.csv"
    if not climate.exists():
        return {"status": "unavailable", "reason": "climate stage did not produce input"}
    with climate.open() as stream:
        rows = list(csv.DictReader(stream))
    args = (5., 30., jnp.asarray([float(r["tmin"]) for r in rows], jnp.float64),
            jnp.asarray([float(r["tmax"]) for r in rows], jnp.float64), 3)
    start = time.perf_counter()
    lowered = degrees_day.lower(*args)
    lowering = time.perf_counter() - start
    start = time.perf_counter()
    compiled = lowered.compile()
    compilation = time.perf_counter() - start
    start = time.perf_counter()
    result = jax.block_until_ready(compiled(*args))
    first = time.perf_counter() - start
    samples = []
    for _ in range(7):
        start = time.perf_counter()
        result = jax.block_until_ready(compiled(*args))
        samples.append(time.perf_counter() - start)
    return {"scope": "GDD kernel only; not equivalent to the full Fortran simulation",
            "dtype": "float64", "shape": [len(rows)], "lowering_seconds": lowering,
            "compilation_seconds": compilation, "first_execution_seconds": first,
            "steady_state_samples_seconds": samples,
            "steady_state_median_seconds": statistics.median(samples),
            "output_sha256": hashlib.sha256(result.tobytes()).hexdigest(),
            "parameters": {"tbase": 5., "tupper": 30., "method": 3}}


def failure_gate(path, environment):
    save(path / "environment.json", environment)
    git_snapshot(path)
    summary = {"parity": False, "steps": {}, "benchmark_comparable": False}
    summary["steps"]["pytest"] = capture([PYTHON, "-m", "pytest", "-q"], path / "pytest-full.log")
    project_list = REFERENCE / "testcase/LIST/ListProjects.txt"
    if project_list.exists():
        for stage in ("climate", "water", "water-evap", "water-crop"):
            summary["steps"][stage] = capture(
                [PYTHON, "-m", "aquacrop_jax.cli", "run", str(project_list),
                 "--stage", stage, "--output", str(path / f"{stage}.csv")], path / f"stage-{stage}.log")
        daily = REFERENCE / "testcase/OUTP_REF/OttawaPRMday.OUT"
        if (path / "water-crop.csv").exists() and daily.exists():
            summary["steps"]["compare-trace"] = capture(
                [PYTHON, "-m", "aquacrop_jax.cli", "compare-trace", str(path / "water-crop.csv"),
                 str(daily), "--output", str(path / "comparison.json")], path / "compare-trace.log")
        try:
            save(path / "benchmark-jax.json", benchmark_jax(path))
        except Exception as exc:
            save(path / "benchmark-jax.json", {"status": "failed", "reason": str(exc)})
    else:
        summary["stages_unavailable"] = f"Missing {project_list}"

    executable = REFERENCE / "src/aquacrop"
    if executable.is_file() and os.access(executable, os.X_OK) and project_list.exists():
        work = path / "fortran"
        shutil.copytree(REFERENCE / "testcase", work)
        (work / "OUTP").mkdir(exist_ok=True)
        for output in (work / "OUTP").glob("*.OUT"):
            output.unlink()  # Only the isolated copy; never touch reference outputs.
        first = capture([str(executable)], path / "fortran-first.log", cwd=work)
        samples, hashes = [], [checksums(work / "OUTP")]
        if first["exit_code"] == 0:
            for index in range(3):
                sample = capture([str(executable)], path / f"fortran-steady-{index}.log", cwd=work)
                samples.append(sample)
                hashes.append(checksums(work / "OUTP"))
        times = [s["seconds"] for s in samples if s["exit_code"] == 0]
        save(path / "benchmark-fortran.json", {"scope": "Full legacy Ottawa simulation",
             "command": [str(executable)], "cwd": str(work), "first_execution": first,
             "steady_state": samples, "steady_state_median_seconds": statistics.median(times) if times else None,
             "compilation_seconds": None, "compilation_note": "Prebuilt immutable executable",
             "dtype": "not probed; prebuilt Fortran reference",
             "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
             "output_sha256": hashes, "repeatable": bool(hashes[0]) and all(h == hashes[0] for h in hashes)})
        if first["exit_code"] == 0 and hashes[0]:
            summary["steps"]["compare-fortran"] = capture(
                [PYTHON, "-m", "aquacrop_jax.cli", "compare", str(work / "OUTP"),
                 str(REFERENCE / "testcase/OUTP_REF")], path / "compare-fortran.log")
    else:
        summary["fortran_unavailable"] = "Reference executable or testcase unavailable"
    save(path / "gate-summary.json", summary)
    targeted = (path / "targeted.status.json").exists()
    reason = ("Test mirato fallito: leggere targeted.status.json e targeted.log. " if targeted else
              "Gate diagnostico manuale: leggere gate-summary.json e i log raccolti. ")
    next_run(path, reason + "Il gate ha raccolto le verifiche disponibili; eventuali scarti "
             "del confronto sono diagnostici, non prova della causa del fallimento.",
             "Isolare la causa dal primo errore in targeted.log, prima di applicare una correzione minima."
             if targeted else "Esaminare gli exit code in gate-summary.json e isolare la prima divergenza Fortran/JAX.")
    # Cloud review does not commit. Capture the checkpoint, including sources, above.
    print(f"Failure gate completato: {path}", flush=True)
    return 0


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Delega locale: redirect HTTP rifiutato")


def delegate(args, environment):
    path = dossier("local")
    save(path / "environment.json", environment)
    (path / "task.txt").write_text(args.task + "\n")
    model = os.environ.get("AQUACROP_LOCAL_MODEL", "")
    endpoint = os.environ.get("AQUACROP_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1") or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
        raise RuntimeError("AQUACROP_OLLAMA_URL deve indicare un endpoint HTTP loopback senza credenziali o percorso")
    chunks = []
    for name in args.file:
        parts = name.rsplit(":", 2)
        ranged = len(parts) == 3 and all(part.isdigit() for part in parts[1:])
        source = (ROOT / (parts[0] if ranged else name)).resolve()
        if not source.is_relative_to(ROOT) and not source.is_relative_to(REFERENCE.resolve()):
            raise RuntimeError(f"Contesto esterno al progetto rifiutato: {name}")
        if ranged:
            first, last = map(int, parts[1:])
            if first < 1 or last < first:
                raise RuntimeError(f"Intervallo di righe non valido: {name}")
            with source.open() as stream:
                content = "".join(f"{index}: {line}" for index, line in enumerate(stream, 1)
                                  if first <= index <= last)
            if not content:
                raise RuntimeError(f"Intervallo senza contenuto: {name}")
        else:
            if source.stat().st_size > 128 * 1024:
                raise RuntimeError(f"File troppo grande: usare percorso:riga_iniziale:riga_finale: {name}")
            content = source.read_text()
        chunks.append(f"FILE {name}\n{content}\nEND FILE\n")
    context = "\n".join(chunks)
    if len(context.encode()) > 128 * 1024:
        raise RuntimeError("Contesto locale oltre 128 KiB: restringere l'unità di lavoro")
    # No proxies or redirects: this request must never leave loopback.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(endpoint + "/api/tags", timeout=5) as response:
            installed = [entry["name"] for entry in json.load(response).get("models", [])]
        if not model:
            raise RuntimeError("AQUACROP_LOCAL_MODEL non configurato. Modelli locali disponibili: " + ", ".join(installed))
        if model not in installed:
            raise RuntimeError(f"Modello {model!r} non presente localmente; nessun download automatico")
        prompt = ("Una sola unità di lavoro. Read-only: non eseguire comandi, non modificare file, "
                  "non richiamare cloud o agenti. Tutti i processi numerici devono restare CPU-only "
                  "e float64. Analizza soltanto il contesto fornito; indica le prove mancanti. "
                  "Fornisci al massimo 20 righe, con percorsi e simboli.\n\n" + args.task + "\n\n" + context)
        request = urllib.request.Request(endpoint + "/api/generate", data=json.dumps(
            {"model": model, "prompt": prompt, "stream": False, "keep_alive": 0,
             "options": {"num_gpu": 0, "num_thread": int(os.environ["AQUACROP_CPU_THREADS"]),
                         "num_predict": 1024, "temperature": 0}}).encode(),
            headers={"Content-Type": "application/json"})
        with opener.open(request, timeout=900) as response:
            result = json.load(response)
        if result.get("error") or not result.get("done") or not result.get("response"):
            raise RuntimeError(f"Risposta locale incompleta: {result.get('error', 'nessun testo finale')}")
        save(path / "response.json", result)
        (path / "response.txt").write_text(result["response"] + "\n")
        save(path / "status.json", {"exit_code": 0, "model": model, "endpoint": endpoint, "num_gpu": 0})
        print(result["response"])
        print(f"Dossier locale: {path}")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        reason = f"{type(exc).__name__}: {exc}"
        save(path / "status.json", {"exit_code": 69, "model": model, "endpoint": endpoint, "reason": reason})
        next_run(path, "Delega locale non eseguita: " + reason,
                 "Rendere disponibile Ollama sul loopback e configurare AQUACROP_LOCAL_MODEL "
                 "con un modello già installato, poi riprendere task.txt una sola volta.")
        print(reason, file=sys.stderr)
        print(f"Dossier locale: {path}", file=sys.stderr)
        return 69


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    test = sub.add_parser("test-and-capture", help="pytest args, or -- followed by an explicit command")
    test.add_argument("command", nargs=argparse.REMAINDER)
    gate = sub.add_parser("failure-gate")
    gate.add_argument("--dossier", type=Path)
    local = sub.add_parser("delegate-local", help="bounded read-only local review; never applies patches")
    local.add_argument("task")
    local.add_argument("--file", action="append", default=[], help="path or path:first:last; repeat for each context file")
    if sys.argv[1:2] == ["test-and-capture"] and sys.argv[2:] != ["--help"]:
        args = parser.parse_args(["test-and-capture"])
        args.command = sys.argv[2:]
    else:
        args = parser.parse_args()
    environment = preflight()
    print(f"CPU-only: {environment['devices']}; float64", flush=True)
    if args.action == "test-and-capture":
        path = dossier("test")
        save(path / "environment.json", environment)
        command = args.command
        if command and command[0] == "--":
            command = command[1:]
            if not command:
                parser.error("-- richiede un comando")
        else:
            command = [PYTHON, "-m", "pytest", *command]
        status = capture(command, path / "targeted.log")
        if status["exit_code"]:
            for line in (path / "targeted.log").read_text(errors="replace").splitlines()[-20:]:
                print(line)
            gate_status = subprocess.run([str(ROOT / "tools/failure-gate"), "--dossier", str(path)], cwd=ROOT)
            save(path / "failure-gate.status.json", {"exit_code": gate_status.returncode})
        print(f"Dossier test: {path}", flush=True)
        return status["exit_code"] if status["exit_code"] >= 0 else 128 - status["exit_code"]
    if args.action == "failure-gate":
        path = args.dossier.resolve() if args.dossier else dossier("gate")
        if not path.is_relative_to((ROOT / ".sequence3").resolve()):
            parser.error("Il dossier deve essere interno a .sequence3")
        path.mkdir(parents=True, exist_ok=True)
        return failure_gate(path, environment)
    lock_path = ROOT / ".sequence3" / "delegate-local.lock"
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Delega locale già in corso; esecuzione parallela rifiutata")
        return delegate(args, environment)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(78)
