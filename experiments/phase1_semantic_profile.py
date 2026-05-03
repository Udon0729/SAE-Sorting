"""Phase 1 cheap semantic profile (dataset.md §8 Phase 1).

For every (prompt, layer, feature) where the feature fires (Top-K + ReLU > 0)
during a forward pass of D_probe_v001 small through Pythia-160m-deduped, we
record max-over-tokens activation. We then aggregate to per (layer, feature):

  - n_firings_total
  - overall_firing_rate                   (n_firings / n_prompts)
  - mean_value, max_value_observed
  - per-category n_firings (C1-C4)
  - per-category firing_rate
  - selectivity_max_over_mean             (Phase 2 candidate selection input)
  - top_qids / top_values                 (top-20 activating prompts)

Outputs:
  data/profiling/v001/phase1_firings.parquet   (raw per-prompt-layer-feature)
  data/profiling/v001/phase1_features.parquet  (summary per layer-feature)
  configs/hash_log.json updated with phase1_features_hash

Phase 1 scope (intentional):
  - All 12 Pythia-160m layers (residual stream output of each block)
  - Full 65,536 SAE features per layer
  - All 1,200 D_probe_v001 small prompts
  - max-over-tokens per (prompt, layer, feature); per-token-position
    decomposition deferred to Phase 3 (light causal)

Run via scheduler (preferred):
  command:        uv run python experiments/phase1_semantic_profile.py
  requested_gpus: [0]
  vram_budget_gib: 8
  env: { SCHEDULER_JOB_ID: "<job_id>" }
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import safetensors.torch as st_torch
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, GPTNeoXForCausalLM

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_FILE = REPO_ROOT / "data" / "probe" / "v001" / "prompts.jsonl"
OUT_DIR = REPO_ROOT / "data" / "profiling" / "v001"
CONFIGS = REPO_ROOT / "configs"
HASH_LOG = CONFIGS / "hash_log.json"

MODEL_REPO = "EleutherAI/pythia-160m-deduped"
MODEL_REVISION = "step143000"
SAE_REPO = "EleutherAI/sae-pythia-160m-deduped-32k"
DEVICE = "cuda:0"
DTYPE = torch.float32
MAX_TOKENS = 64
TOP_K_QIDS_PER_FEATURE = 20


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pythia():
    tok = AutoTokenizer.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
    model = GPTNeoXForCausalLM.from_pretrained(
        MODEL_REPO, revision=MODEL_REVISION, torch_dtype=DTYPE,
    ).to(DEVICE).eval()
    return tok, model


def load_all_saes(n_layers: int) -> list[dict]:
    saes = []
    for L in range(n_layers):
        cfg = json.loads(Path(hf_hub_download(SAE_REPO, f"layers.{L}/cfg.json")).read_text())
        weights = st_torch.load_file(hf_hub_download(SAE_REPO, f"layers.{L}/sae.safetensors"), device=DEVICE)
        weights = {k: v.to(DTYPE) for k, v in weights.items()}
        saes.append({"cfg": cfg, "weights": weights})
    return saes


def sae_encode_topk(h: torch.Tensor, sae: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """h: (T, d_in) -> (indices (T, k), values (T, k)) with ReLU on values when signed=False."""
    w = sae["weights"]
    cfg = sae["cfg"]
    sae_in = h - w["b_dec"]
    pre_acts = sae_in @ w["encoder.weight"].T + w["encoder.bias"]
    topk = pre_acts.topk(int(cfg["k"]), dim=-1)
    values = topk.values
    if not bool(cfg.get("signed", False)):
        values = values.relu()
    return topk.indices, values


def reduce_per_feature_max(indices: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given parallel arrays (n,) of feature indices and values, return per-feature max.
    Vectorized via sort + np.maximum.reduceat."""
    if indices.size == 0:
        return np.array([], dtype=indices.dtype), np.array([], dtype=values.dtype)
    order = np.argsort(indices, kind="stable")
    sorted_idx = indices[order]
    sorted_val = values[order]
    unique_idx, start_pos = np.unique(sorted_idx, return_index=True)
    max_vals = np.maximum.reduceat(sorted_val, start_pos)
    return unique_idx, max_vals


def collect_firings(tok, model, saes: list[dict], prompts: list[dict]) -> pd.DataFrame:
    n_layers = len(saes)
    rows: list[dict] = []
    t0 = time.time()
    for i, p in enumerate(prompts):
        if i % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            print(f"  [{i:4d}/{len(prompts)}] {elapsed:6.1f}s elapsed ({rate:.1f} prompts/s)")
        inputs = tok(p["prompt"], return_tensors="pt", truncation=True, max_length=MAX_TOKENS).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        for L in range(n_layers):
            h = outputs.hidden_states[L + 1][0]  # (T, d_in)
            indices, values = sae_encode_topk(h, saes[L])
            mask = values > 0
            if not mask.any():
                continue
            flat_idx = indices[mask].to(torch.int32).cpu().numpy()
            flat_val = values[mask].cpu().numpy()
            uniq_idx, max_vals = reduce_per_feature_max(flat_idx, flat_val)
            for fidx, mv in zip(uniq_idx, max_vals):
                rows.append({
                    "qid": p["qid"],
                    "category": p["category"],
                    "prompt_type": p["prompt_type"],
                    "layer": L,
                    "feature_idx": int(fidx),
                    "max_value": float(mv),
                })
    print(f"  done: {len(rows)} (prompt, layer, feature) firings in {time.time() - t0:.1f}s")
    return pd.DataFrame(rows)


def aggregate_summary(df: pd.DataFrame, cat_totals: dict[str, int]) -> pd.DataFrame:
    """Aggregate raw firings into per (layer, feature) summary."""
    n_total = sum(cat_totals.values())

    # Main stats
    main = df.groupby(["layer", "feature_idx"]).agg(
        n_firings_total=("qid", "count"),
        mean_value=("max_value", "mean"),
        max_value_observed=("max_value", "max"),
    ).reset_index()
    main["overall_firing_rate"] = main["n_firings_total"] / n_total

    # Per-category counts (pivot)
    cat_n = df.groupby(["layer", "feature_idx", "category"]).size().unstack(fill_value=0)
    cat_n.columns = [f"n_firings_{c}" for c in cat_n.columns]
    cat_n = cat_n.reset_index()

    cat_cols = [c for c in cat_n.columns if c.startswith("n_firings_")]
    rate_cols: list[str] = []
    for col in cat_cols:
        cat_name = col[len("n_firings_"):]
        total = cat_totals.get(cat_name, 0)
        rate_col = f"firing_rate_{cat_name}"
        if total > 0:
            cat_n[rate_col] = cat_n[col] / total
        else:
            cat_n[rate_col] = 0.0
        rate_cols.append(rate_col)

    rate_matrix = cat_n[rate_cols].to_numpy()
    mean_rate = rate_matrix.mean(axis=1)
    max_rate = rate_matrix.max(axis=1)
    cat_n["selectivity_max_over_mean"] = max_rate / (mean_rate + 1e-9)

    # Top-K activating prompts (top_qids / top_values)
    top = (
        df.sort_values(["layer", "feature_idx", "max_value"], ascending=[True, True, False])
          .groupby(["layer", "feature_idx"])
          .agg(
              top_qids=("qid", lambda s: s.head(TOP_K_QIDS_PER_FEATURE).tolist()),
              top_values=("max_value", lambda s: s.head(TOP_K_QIDS_PER_FEATURE).tolist()),
          )
          .reset_index()
    )

    summary = main.merge(cat_n, on=["layer", "feature_idx"], how="left")
    summary = summary.merge(top, on=["layer", "feature_idx"], how="left")
    return summary


def write_torch_peak(job_id: str | None) -> None:
    if not job_id:
        return
    target = REPO_ROOT / "ops" / "queue" / "logs" / f"{job_id}_torch_peak.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"0": int(torch.cuda.max_memory_allocated(0))}))


def update_hash_log(summary_path: Path, n_layers: int, n_features: int, n_prompts: int) -> None:
    log = json.loads(HASH_LOG.read_text()) if HASH_LOG.exists() else {}
    log["phase1_features_hash"] = {
        "file": str(summary_path.relative_to(REPO_ROOT)),
        "sha256": sha256_of(summary_path),
        "n_layers": n_layers,
        "n_features": n_features,
        "n_prompts": n_prompts,
        "model": f"{MODEL_REPO}@{MODEL_REVISION}",
        "sae": SAE_REPO,
        "frozen_at": now_iso(),
    }
    HASH_LOG.write_text(json.dumps(log, indent=2, ensure_ascii=False, sort_keys=True))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"--- Phase 1 semantic profile ---")
    print(f"  device: {torch.cuda.get_device_name(0)}")
    print(f"  torch:  {torch.__version__}")

    print("[load] Pythia + SAEs (12 layers)")
    tok, model = load_pythia()
    n_layers = model.config.num_hidden_layers
    saes = load_all_saes(n_layers)
    print(f"  n_layers={n_layers}, all SAEs loaded into VRAM")
    print(f"  param VRAM after load: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GiB")

    print("[load] D_probe_v001 small")
    prompts = [json.loads(l) for l in PROMPT_FILE.read_text().splitlines()]
    cat_totals = pd.Series([p["category"] for p in prompts]).value_counts().to_dict()
    print(f"  n_prompts={len(prompts)}, by category: {cat_totals}")

    print("[forward+SAE] collecting firings")
    df = collect_firings(tok, model, saes, prompts)
    raw_path = OUT_DIR / "phase1_firings.parquet"
    df.to_parquet(raw_path, index=False)
    print(f"  raw saved: {raw_path} ({len(df)} rows)")

    print("[aggregate] per (layer, feature) summary")
    summary = aggregate_summary(df, cat_totals)
    summary_path = OUT_DIR / "phase1_features.parquet"
    summary.to_parquet(summary_path, index=False)
    print(f"  summary saved: {summary_path} ({len(summary)} unique (layer, feature))")

    update_hash_log(summary_path, n_layers, len(summary), len(prompts))

    # Print top-10 most-fired (layer, feature) for sanity
    print()
    print("[top-10 most-fired (layer, feature)]")
    top10 = summary.nlargest(10, "n_firings_total")[
        ["layer", "feature_idx", "n_firings_total", "selectivity_max_over_mean", "max_value_observed"]
    ]
    print(top10.to_string(index=False))

    print()
    print("[per-layer (layer, feature) counts]")
    print(summary["layer"].value_counts().sort_index().to_string())

    print()
    print(f"[final VRAM peak] {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GiB")
    write_torch_peak(os.environ.get("SCHEDULER_JOB_ID"))
    print(f"[done]")


if __name__ == "__main__":
    main()
