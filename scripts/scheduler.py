"""SAE-Sorting GPU job scheduler — Phase A + Phase B.

One tick:
  1. Finalize completed jobs (frees GPU reservations).
  2. Query GPU state and launch eligible pending jobs (FIFO by job_id).
  3. Append finalized jobs to ops/queue/history/job_history.parquet.

Phase A scope:
  - 4 state directories (pending/running/completed/failed)
  - nohup + start_new_session detached launch
  - kill -0 (os.kill(pid, 0)) liveness polling
  - exit_code captured by inline bash wrapper
  - job_history aggregation

Phase B additions:
  - GPU availability detection via `nvidia-smi --query-gpu=...`
      free GPU = mem_used/mem_total < 0.10 AND util_gpu < 5 AND not reserved
      reservation = union of running jobs' assigned_gpus
  - requested_gpus: list (explicit IDs) or int (auto-assign N)
  - vram_budget_gib: optional check against GPU's free memory
  - System-level VRAM peak via `nvidia-smi dmon -s mu -i {gpus} -d 5` background process
      CSV at ops/queue/logs/{job_id}_vram_dmon.csv; max(fb_mib) per GPU at finalize
  - Per-process VRAM peak via {job_id}_torch_peak.json (user code writes)
      schema: {"<gpu_idx>": <bytes>} or {"cuda:<idx>": <bytes>}
  - Dual-system divergence >= 5 GiB → warning printed at finalize

Out of scope (Phase C):
  - depends_on auto-resolution
  - co_locatable multi-tenant on same GPU
  - retry policy
"""

from __future__ import annotations

import json
import os
import signal
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
GPU_FREE_MEM_FRAC = 0.10
GPU_FREE_UTIL = 5
VRAM_DIVERGENCE_WARN_GIB = 5.0
MIB_PER_GIB = 1024.0


# -- helpers (Phase A) --------------------------------------------------------

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


# -- Phase B: GPU detection ---------------------------------------------------

def query_gpus() -> list[dict[str, Any]]:
    """Query nvidia-smi for GPU state. Returns [] if nvidia-smi unavailable."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            gpus.append({
                "index": int(parts[0]),
                "mem_used_mib": int(parts[1]),
                "mem_total_mib": int(parts[2]),
                "util_gpu": int(parts[3]),
            })
        except ValueError:
            continue
    return gpus


def collect_reserved_gpus() -> set[int]:
    """GPU indices currently reserved by running jobs."""
    reserved: set[int] = set()
    for spec_path in RUNNING.glob("*.yaml"):
        try:
            spec = load_yaml(spec_path)
        except Exception:
            continue
        for g in spec.get("assigned_gpus") or []:
            try:
                reserved.add(int(g))
            except (TypeError, ValueError):
                continue
    return reserved


def select_gpus_for_job(
    spec: dict[str, Any],
    gpu_state: list[dict[str, Any]],
    reserved: set[int],
) -> list[int] | None:
    """Pick GPU indices for a job. Returns:
      []    -> CPU job, no GPUs needed
      [...] -> assigned GPU indices
      None  -> requirements cannot be met (skip this tick)
    """
    requested = spec.get("requested_gpus")
    if not requested:  # None, 0, [], False
        return []
    if not gpu_state:
        # nvidia-smi unavailable: cannot validate GPU jobs, defer
        return None

    vram_budget_mib = int(float(spec.get("vram_budget_gib") or 0) * MIB_PER_GIB)
    by_idx = {g["index"]: g for g in gpu_state}

    def is_free(idx: int) -> bool:
        if idx in reserved:
            return False
        g = by_idx.get(idx)
        if g is None:
            return False
        if g["mem_total_mib"] > 0 and g["mem_used_mib"] / g["mem_total_mib"] >= GPU_FREE_MEM_FRAC:
            return False
        if g["util_gpu"] >= GPU_FREE_UTIL:
            return False
        if vram_budget_mib > 0 and (g["mem_total_mib"] - g["mem_used_mib"]) < vram_budget_mib:
            return False
        return True

    if isinstance(requested, list):
        if all(is_free(int(idx)) for idx in requested):
            return [int(idx) for idx in requested]
        return None
    if isinstance(requested, int):
        free = sorted(g["index"] for g in gpu_state if is_free(g["index"]))
        if len(free) < requested:
            return None
        return free[:requested]
    return None


# -- Phase B: VRAM measurement -----------------------------------------------

def start_vram_dmon(job_id: str, gpus: list[int]) -> int | None:
    """Start `nvidia-smi dmon -s mu -i {gpus} -o T -d 5` as background process.
    Returns dmon PID or None on failure / no GPUs."""
    if not gpus:
        return None
    csv_path = LOGS / f"{job_id}_vram_dmon.csv"
    cmd = [
        "nvidia-smi", "dmon", "-s", "mu",
        "-i", ",".join(str(g) for g in gpus),
        "-o", "T", "-d", "5",
    ]
    try:
        f = csv_path.open("w")
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        return proc.pid
    except FileNotFoundError:
        return None


def stop_dmon(monitor_pid: int | None) -> None:
    if monitor_pid is None:
        return
    try:
        os.kill(int(monitor_pid), signal.SIGTERM)
    except (ProcessLookupError, ValueError, TypeError):
        pass


def parse_dmon_csv(path: Path) -> dict[int, int]:
    """Parse nvidia-smi dmon CSV. Returns {gpu_index: peak_fb_mib}.
    Format (from `dmon -s mu -o T -d N`):
      #Time         gpu     fb   bar1   ccpm     sm    mem    enc    dec    jpg    ofa
      #HH:MM:SS     Idx     MB     MB     MB      %      %      %      %      %      %
       12:08:12       0      1      2      0      0      0      0      0      0      0
    """
    if not path.exists():
        return {}
    peaks: dict[int, int] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            gpu_idx = int(parts[1])
            fb_mib = int(parts[2])
        except ValueError:
            continue
        peaks[gpu_idx] = max(peaks.get(gpu_idx, 0), fb_mib)
    return peaks


def parse_torch_peak(job_id: str) -> dict[int, int] | None:
    """Read {job_id}_torch_peak.json if user code wrote it.
    Accepts {"0": bytes} or {"cuda:0": bytes} or {"gpu_0": bytes}."""
    path = LOGS / f"{job_id}_torch_peak.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    out: dict[int, int] = {}
    for k, v in data.items():
        s = str(k).replace("cuda:", "").replace("gpu_", "")
        try:
            out[int(s)] = int(v)
        except (ValueError, TypeError):
            continue
    return out or None


def vram_summary(
    job_id: str,
    assigned_gpus: list[int],
) -> dict[str, Any]:
    """Compute per-GPU VRAM peaks from both sources and warn on divergence."""
    out: dict[str, Any] = {
        "vram_peak_dmon_mib": {},
        "vram_peak_dmon_gib": {},
        "vram_peak_torch_bytes": None,
        "vram_peak_torch_gib": None,
        "vram_divergence_warning": [],
    }
    if not assigned_gpus:
        return out

    dmon_peaks_mib = parse_dmon_csv(LOGS / f"{job_id}_vram_dmon.csv")
    out["vram_peak_dmon_mib"] = {str(k): v for k, v in dmon_peaks_mib.items()}
    out["vram_peak_dmon_gib"] = {str(k): round(v / MIB_PER_GIB, 3) for k, v in dmon_peaks_mib.items()}

    torch_peaks = parse_torch_peak(job_id)
    if torch_peaks is not None:
        out["vram_peak_torch_bytes"] = {str(k): v for k, v in torch_peaks.items()}
        out["vram_peak_torch_gib"] = {
            str(k): round(v / (MIB_PER_GIB * MIB_PER_GIB * MIB_PER_GIB), 3)  # bytes -> GiB
            for k, v in torch_peaks.items()
        }
        for idx in assigned_gpus:
            t = torch_peaks.get(idx)
            d = dmon_peaks_mib.get(idx)
            if t is None or d is None:
                continue
            t_gib = t / (1024.0 ** 3)
            d_gib = d / MIB_PER_GIB
            diff = abs(t_gib - d_gib)
            if diff >= VRAM_DIVERGENCE_WARN_GIB:
                out["vram_divergence_warning"].append({
                    "gpu_index": idx,
                    "torch_gib": round(t_gib, 3),
                    "dmon_gib": round(d_gib, 3),
                    "diff_gib": round(diff, 3),
                })
    return out


# -- env + launch (Phase A + B) -----------------------------------------------

def build_env(spec: dict[str, Any], assigned_gpus: list[int]) -> dict[str, str]:
    env = os.environ.copy()
    if assigned_gpus:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in assigned_gpus)
    env["PYTHONHASHSEED"] = str(spec.get("pin_seed", 0))
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    for k, v in (spec.get("env") or {}).items():
        env[str(k)] = str(v)
    return env


def launch_job(spec_path: Path, assigned_gpus: list[int]) -> None:
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
        env=build_env(spec, assigned_gpus),
        cwd=str(REPO_ROOT),
    )
    pid = proc.pid
    (PIDS / f"{job_id}.pid").write_text(f"{pid}\n")

    monitor_pid = start_vram_dmon(job_id, assigned_gpus)

    spec["pid"] = pid
    spec["start_time"] = now_iso()
    spec["status"] = "running"
    spec["assigned_gpus"] = assigned_gpus
    spec["monitor_pid"] = monitor_pid

    write_yaml(RUNNING / f"{job_id}.yaml", spec)
    spec_path.unlink()
    print(f"[launch] {job_id} pid={pid} gpus={assigned_gpus} dmon_pid={monitor_pid}")


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

    # Stop dmon (job is done; dmon should exit)
    stop_dmon(spec.get("monitor_pid"))

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

    # Phase B: VRAM peaks
    vram = vram_summary(job_id, spec.get("assigned_gpus") or [])
    spec.update(vram)
    if vram["vram_divergence_warning"]:
        for w in vram["vram_divergence_warning"]:
            print(f"[warn vram] {job_id} gpu={w['gpu_index']} torch={w['torch_gib']}GiB "
                  f"dmon={w['dmon_gib']}GiB diff={w['diff_gib']}GiB")

    if exit_code == 0:
        spec["status"] = "completed"
        dst = COMPLETED / f"{job_id}.yaml"
    else:
        spec["status"] = "failed"
        spec["error_tail"] = read_error_tail(job_id)
        dst = FAILED / f"{job_id}.yaml"

    write_yaml(dst, spec)
    running_path.unlink()
    print(f"[finalize] {job_id} status={spec['status']} exit_code={exit_code} duration={duration_sec}s "
          f"vram_dmon={vram['vram_peak_dmon_gib']}")
    return spec


def append_history(spec: dict[str, Any]) -> None:
    row = {
        "job_id": spec.get("job_id"),
        "command": spec.get("command"),
        "requested_gpus": str(spec.get("requested_gpus") or []),
        "assigned_gpus": str(spec.get("assigned_gpus") or []),
        "pid": spec.get("pid"),
        "monitor_pid": spec.get("monitor_pid"),
        "start_time": spec.get("start_time"),
        "end_time": spec.get("end_time"),
        "duration_sec": spec.get("duration_sec"),
        "exit_code": spec.get("exit_code"),
        "status": spec.get("status"),
        "vram_peak_dmon_gib": json.dumps(spec.get("vram_peak_dmon_gib") or {}),
        "vram_peak_torch_gib": json.dumps(spec.get("vram_peak_torch_gib") or {}),
    }
    df_new = pd.DataFrame([row])
    if HISTORY.exists():
        df = pd.concat([pd.read_parquet(HISTORY), df_new], ignore_index=True)
    else:
        df = df_new
    df.to_parquet(HISTORY, index=False)


# -- main loop ---------------------------------------------------------------

def tick() -> None:
    ensure_dirs()

    # 1. Finalize completed jobs first (frees GPU reservations for this tick)
    for spec_path in sorted(RUNNING.glob("*.yaml")):
        try:
            finalized = finalize_job(spec_path)
            if finalized is not None:
                append_history(finalized)
        except Exception as e:
            print(f"[error finalize] {spec_path.name}: {e}")

    # 2. Launch eligible pending jobs (FIFO, gated by GPU availability)
    gpu_state = query_gpus()
    reserved = collect_reserved_gpus()
    for spec_path in sorted(PENDING.glob("*.yaml")):
        try:
            spec = load_yaml(spec_path)
            assigned = select_gpus_for_job(spec, gpu_state, reserved)
            if assigned is None:
                # Requirements not met (no free GPU or nvidia-smi unavailable for GPU job)
                continue
            for g in assigned:
                reserved.add(g)
            launch_job(spec_path, assigned_gpus=assigned)
        except Exception as e:
            print(f"[error launch] {spec_path.name}: {e}")


if __name__ == "__main__":
    tick()
