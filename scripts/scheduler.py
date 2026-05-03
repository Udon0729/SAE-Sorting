"""SAE-Sorting GPU job scheduler — Phase A.

One tick:
  1. Launch all pending jobs (FIFO by job_id) via nohup + setsid.
  2. Check running jobs: if PID dead, finalize to completed/ or failed/.
  3. Append finalized jobs to ops/queue/history/job_history.parquet.

Phase A scope:
  - 4 state directories (pending/running/completed/failed)
  - nohup + start_new_session detached launch
  - kill -0 (os.kill(pid, 0)) liveness polling
  - exit_code captured by inline bash wrapper
  - job_history aggregation

Out of scope (Phase B/C):
  - GPU availability detection (nvidia-smi parse)
  - VRAM peak measurement (torch / nvidia-smi dmon)
  - depends_on auto-resolution
  - co_locatable multi-tenant
  - retry
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
QUEUE = REPO_ROOT / "ops" / "queue"
PENDING = QUEUE / "pending"
RUNNING = QUEUE / "running"
COMPLETED = QUEUE / "completed"
FAILED = QUEUE / "failed"
PIDS = QUEUE / "pids"
EXIT_CODES = QUEUE / "exit_codes"
LOGS = QUEUE / "logs"
HISTORY_DIR = QUEUE / "history"
HISTORY = HISTORY_DIR / "job_history.parquet"

ERROR_TAIL_BYTES = 4096
ERROR_TAIL_LINES = 50


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def ensure_dirs() -> None:
    for d in (PENDING, RUNNING, COMPLETED, FAILED, PIDS, EXIT_CODES, LOGS, HISTORY_DIR):
        d.mkdir(parents=True, exist_ok=True)


def build_env(spec: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    requested_gpus = spec.get("requested_gpus") or []
    if requested_gpus:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in requested_gpus)
    env["PYTHONHASHSEED"] = str(spec.get("pin_seed", 0))
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    for k, v in (spec.get("env") or {}).items():
        env[str(k)] = str(v)
    return env


def launch_job(spec_path: Path) -> None:
    spec = load_yaml(spec_path)
    job_id = spec["job_id"]
    command = spec["command"]

    exit_code_path = EXIT_CODES / f"{job_id}.code"
    exit_code_path.unlink(missing_ok=True)

    inner = f"({command})\necho $? > {exit_code_path}\n"

    stdout_f = (LOGS / f"{job_id}.stdout").open("ab")
    stderr_f = (LOGS / f"{job_id}.stderr").open("ab")

    proc = subprocess.Popen(
        ["nohup", "bash", "-c", inner],
        stdout=stdout_f,
        stderr=stderr_f,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        env=build_env(spec),
        cwd=str(REPO_ROOT),
    )
    pid = proc.pid
    (PIDS / f"{job_id}.pid").write_text(f"{pid}\n")

    spec["pid"] = pid
    spec["start_time"] = now_iso()
    spec["status"] = "running"
    spec["assigned_gpus"] = spec.get("requested_gpus") or []

    write_yaml(RUNNING / f"{job_id}.yaml", spec)
    spec_path.unlink()
    print(f"[launch] {job_id} pid={pid} gpus={spec['assigned_gpus']}")


def read_error_tail(job_id: str) -> list[str]:
    stderr_path = LOGS / f"{job_id}.stderr"
    if not stderr_path.exists():
        return []
    try:
        with stderr_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - ERROR_TAIL_BYTES))
            tail = f.read().decode("utf-8", errors="replace")
        return tail.splitlines()[-ERROR_TAIL_LINES:]
    except OSError:
        return ["<could not read stderr>"]


def finalize_job(running_path: Path) -> dict[str, Any] | None:
    spec = load_yaml(running_path)
    job_id = spec["job_id"]
    pid = int(spec["pid"])

    if is_pid_alive(pid):
        return None

    end_time = now_iso()
    duration_sec: float | None = None
    if spec.get("start_time"):
        try:
            t0 = datetime.fromisoformat(spec["start_time"])
            t1 = datetime.fromisoformat(end_time)
            duration_sec = (t1 - t0).total_seconds()
        except ValueError:
            pass

    exit_code_path = EXIT_CODES / f"{job_id}.code"
    if exit_code_path.exists():
        try:
            exit_code = int(exit_code_path.read_text().strip())
        except ValueError:
            exit_code = -1
    else:
        exit_code = -1

    spec["end_time"] = end_time
    spec["duration_sec"] = duration_sec
    spec["exit_code"] = exit_code

    if exit_code == 0:
        spec["status"] = "completed"
        dst = COMPLETED / f"{job_id}.yaml"
    else:
        spec["status"] = "failed"
        spec["error_tail"] = read_error_tail(job_id)
        dst = FAILED / f"{job_id}.yaml"

    write_yaml(dst, spec)
    running_path.unlink()
    print(
        f"[finalize] {job_id} status={spec['status']} "
        f"exit_code={exit_code} duration={duration_sec}s"
    )
    return spec


def append_history(spec: dict[str, Any]) -> None:
    row = {
        "job_id": spec.get("job_id"),
        "command": spec.get("command"),
        "requested_gpus": str(spec.get("requested_gpus") or []),
        "assigned_gpus": str(spec.get("assigned_gpus") or []),
        "pid": spec.get("pid"),
        "start_time": spec.get("start_time"),
        "end_time": spec.get("end_time"),
        "duration_sec": spec.get("duration_sec"),
        "exit_code": spec.get("exit_code"),
        "status": spec.get("status"),
    }
    df_new = pd.DataFrame([row])
    if HISTORY.exists():
        df = pd.concat([pd.read_parquet(HISTORY), df_new], ignore_index=True)
    else:
        df = df_new
    df.to_parquet(HISTORY, index=False)


def tick() -> None:
    ensure_dirs()

    for spec_path in sorted(PENDING.glob("*.yaml")):
        try:
            launch_job(spec_path)
        except Exception as e:
            print(f"[error launch] {spec_path.name}: {e}")

    for spec_path in sorted(RUNNING.glob("*.yaml")):
        try:
            finalized = finalize_job(spec_path)
            if finalized is not None:
                append_history(finalized)
        except Exception as e:
            print(f"[error finalize] {spec_path.name}: {e}")


if __name__ == "__main__":
    tick()
