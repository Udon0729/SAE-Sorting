"""Phase 1 cheap semantic profile for Qwen3-8B-Base + Qwen-Scope SAE.

Mirrors experiments/phase1_semantic_profile.py (Pythia-160m) but adapted for
the Qwen-Scope SAE family:
  - SAE checkpoint format: .pt dict with W_enc/W_dec/b_enc/b_dec (no b_dec
    subtract before encoder, raw top-K scatter without ReLU per their README).
  - 36 layers, 2.15 GiB per SAE -> cannot load all simultaneously.

Strategy:
  Phase 1A: forward all prompts once, cache selected layers' residual stream
            in CPU memory (bf16, ~3.7 GiB for 6 layers x 1200 prompts x 64 tok).
  Phase 1B: per layer, load one SAE -> apply to all cached hidden_states ->
            collect (prompt, layer, feature, max_value) firings -> free.

Layer scope (intentional): 6 evenly-spaced layers (0, 7, 14, 21, 28, 35).
v002 TODO: extend to all 36 layers if results warrant.

Outputs:
  data/profiling/v001/phase1_qwen3_8b_firings.parquet   (raw per-prompt-layer-feature)
  data/profiling/v001/phase1_qwen3_8b_features.parquet  (summary per layer-feature)
  configs/hash_log.json updated with phase1_qwen3_8b_features_hash

Run via scheduler:
  command:        uv run python experiments/phase1_qwen3_8b_semantic_profile.py
  requested_gpus: [0]
  vram_budget_gib: 24
  env:
    SCHEDULER_JOB_ID: "<job_id>"
    HF_HOME: "/diskthalys/.../data/raw/hf_cache"
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
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_FILE = REPO_ROOT / "data" / "probe" / "v001" / "prompts.jsonl"
OUT_DIR = REPO_ROOT / "data" / "profiling" / "v001"
CONFIGS = REPO_ROOT / "configs"
HASH_LOG = CONFIGS / "hash_log.json"

MODEL_REPO = "Qwen/Qwen3-8B-Base"
SAE_REPO = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"
SELECTED_LAYERS = list(range(36))  # all 36 layers (extended from 6 evenly spaced)
DEVICE = "cuda:0"
MODEL_DTYPE = torch.bfloat16
SAE_DTYPE = torch.float32
MAX_TOKENS = 64
SAE_TOP_K = 50
TOP_K_QIDS_PER_FEATURE = 20


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_qwen():
    tok = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = AutoModelForCausalLM.from_pretrained(MODEL_REPO, torch_dtype=MODEL_DTYPE).to(DEVICE).eval()
    return tok, model


def load_sae_layer(layer: int) -> dict[str, torch.Tensor]:
    """Load one SAE checkpoint to GPU in fp32. Returns dict with W_enc, W_dec, b_enc, b_dec."""
    path = hf_hub_download(SAE_REPO, f"layer{layer}.sae.pt")
    sae = torch.load(path, map_location="cpu", weights_only=True)
    return {k: sae[k].to(device=DEVICE, dtype=SAE_DTYPE) for k in ("W_enc", "W_dec", "b_enc", "b_dec")}


def sae_topk_qwen(residual_fp32: torch.Tensor, sae: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen-Scope SAE forward (per their README). residual: (T, d_model) fp32.
    Returns (topk_indices (T, k), topk_values (T, k))."""
    pre_acts = residual_fp32 @ sae["W_enc"].T + sae["b_enc"]  # (T, num_latents)
    topk = pre_acts.topk(SAE_TOP_K, dim=-1)
    return topk.indices, topk.values  # raw values, no ReLU per README


def reduce_per_feature_max(indices: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if indices.size == 0:
        return np.array([], dtype=indices.dtype), np.array([], dtype=values.dtype)
    order = np.argsort(indices, kind="stable")
    sorted_idx = indices[order]
    sorted_val = values[order]
    unique_idx, start_pos = np.unique(sorted_idx, return_index=True)
    return unique_idx, np.maximum.reduceat(sorted_val, start_pos)


def phase1a_forward_cache(tok, model, prompts: list[dict]) -> dict[int, list[torch.Tensor]]:
    """Forward all prompts, cache selected layers' residual stream output in CPU bf16."""
    cache: dict[int, list[torch.Tensor]] = {L: [] for L in SELECTED_LAYERS}
    t0 = time.time()
    for i, p in enumerate(prompts):
        if i % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            print(f"  [forward {i:4d}/{len(prompts)}] {elapsed:6.1f}s ({rate:.1f} prompts/s)")
        inputs = tok(p["prompt"], return_tensors="pt", truncation=True, max_length=MAX_TOKENS).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        for L in SELECTED_LAYERS:
            cache[L].append(outputs.hidden_states[L + 1][0].to("cpu"))
    print(f"  forward done in {time.time() - t0:.1f}s")
    return cache


def phase1b_apply_saes(prompts: list[dict], cache: dict[int, list[torch.Tensor]]) -> pd.DataFrame:
    """For each layer in SELECTED_LAYERS, load SAE -> apply to all cached hiddens -> firings."""
    rows: list[dict] = []
    for L in SELECTED_LAYERS:
        t0 = time.time()
        print(f"  [layer {L}] loading SAE...")
        sae = load_sae_layer(L)
        print(f"    SAE loaded ({(time.time() - t0):.1f}s); applying to {len(prompts)} prompts...")
        t1 = time.time()
        for i, p in enumerate(prompts):
            h = cache[L][i].to(device=DEVICE, dtype=SAE_DTYPE)  # (T, d_model)
            indices, values = sae_topk_qwen(h, sae)
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
        # Free SAE before next layer
        del sae
        torch.cuda.empty_cache()
        print(f"    layer {L} apply done in {(time.time() - t1):.1f}s; total firings so far={len(rows)}")
    return pd.DataFrame(rows)


def aggregate_summary(df: pd.DataFrame, cat_totals: dict[str, int]) -> pd.DataFrame:
    n_total = sum(cat_totals.values())
    main = df.groupby(["layer", "feature_idx"]).agg(
        n_firings_total=("qid", "count"),
        mean_value=("max_value", "mean"),
        max_value_observed=("max_value", "max"),
    ).reset_index()
    main["overall_firing_rate"] = main["n_firings_total"] / n_total

    cat_n = df.groupby(["layer", "feature_idx", "category"]).size().unstack(fill_value=0)
    cat_n.columns = [f"n_firings_{c}" for c in cat_n.columns]
    cat_n = cat_n.reset_index()
    rate_cols: list[str] = []
    for col in [c for c in cat_n.columns if c.startswith("n_firings_")]:
        cat_name = col[len("n_firings_"):]
        total = cat_totals.get(cat_name, 0)
        rate_col = f"firing_rate_{cat_name}"
        cat_n[rate_col] = (cat_n[col] / total) if total > 0 else 0.0
        rate_cols.append(rate_col)
    rate_matrix = cat_n[rate_cols].to_numpy()
    cat_n["selectivity_max_over_mean"] = rate_matrix.max(axis=1) / (rate_matrix.mean(axis=1) + 1e-9)

    top = (
        df.sort_values(["layer", "feature_idx", "max_value"], ascending=[True, True, False])
          .groupby(["layer", "feature_idx"])
          .agg(top_qids=("qid", lambda s: s.head(TOP_K_QIDS_PER_FEATURE).tolist()),
               top_values=("max_value", lambda s: s.head(TOP_K_QIDS_PER_FEATURE).tolist()))
          .reset_index()
    )
    return main.merge(cat_n, on=["layer", "feature_idx"], how="left").merge(top, on=["layer", "feature_idx"], how="left")


def write_torch_peak(job_id: str | None) -> None:
    if not job_id:
        return
    target = REPO_ROOT / "ops" / "queue" / "logs" / f"{job_id}_torch_peak.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"0": int(torch.cuda.max_memory_allocated(0))}))


def update_hash_log(summary_path: Path, n_layers: int, n_features: int, n_prompts: int) -> None:
    log = json.loads(HASH_LOG.read_text()) if HASH_LOG.exists() else {}
    log["phase1_qwen3_8b_features_hash"] = {
        "file": str(summary_path.relative_to(REPO_ROOT)),
        "sha256": sha256_of(summary_path),
        "n_layers_profiled": n_layers,
        "selected_layers": SELECTED_LAYERS,
        "n_features": n_features,
        "n_prompts": n_prompts,
        "model": MODEL_REPO,
        "sae": SAE_REPO,
        "sae_top_k": SAE_TOP_K,
        "frozen_at": now_iso(),
    }
    HASH_LOG.write_text(json.dumps(log, indent=2, ensure_ascii=False, sort_keys=True))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"--- Phase 1 semantic profile (Qwen3-8B-Base) ---")
    print(f"  device: {torch.cuda.get_device_name(0)}")
    print(f"  torch:  {torch.__version__}")
    print(f"  selected layers: {SELECTED_LAYERS}")

    print("[load] Qwen3-8B-Base")
    tok, model = load_qwen()
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"  n_layers={n_layers}, hidden_size={d_model}")
    print(f"  param VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GiB")

    print("[load] D_probe_v001 small")
    prompts = [json.loads(l) for l in PROMPT_FILE.read_text().splitlines()]
    cat_totals = pd.Series([p["category"] for p in prompts]).value_counts().to_dict()
    print(f"  n_prompts={len(prompts)}, by category: {cat_totals}")

    print("[Phase 1A] forward + cache hidden_states for selected layers")
    cache = phase1a_forward_cache(tok, model, prompts)
    cache_bytes = sum(t.element_size() * t.numel() for L in cache for t in cache[L])
    print(f"  CPU cache size: {cache_bytes / 1024**3:.2f} GiB")

    # Free model from GPU before SAEs to maximize headroom
    del model
    torch.cuda.empty_cache()
    print(f"  model freed; VRAM after free: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GiB")

    print("[Phase 1B] streaming SAE per layer")
    df = phase1b_apply_saes(prompts, cache)
    raw_path = OUT_DIR / "phase1_qwen3_8b_firings.parquet"
    df.to_parquet(raw_path, index=False)
    print(f"  raw saved: {raw_path} ({len(df)} firings)")

    print("[aggregate] per (layer, feature) summary")
    summary = aggregate_summary(df, cat_totals)
    summary_path = OUT_DIR / "phase1_qwen3_8b_features.parquet"
    summary.to_parquet(summary_path, index=False)
    print(f"  summary saved: {summary_path} ({len(summary)} unique (layer, feature))")

    update_hash_log(summary_path, len(SELECTED_LAYERS), len(summary), len(prompts))

    print()
    print("[per-layer (layer, feature) counts]")
    print(summary["layer"].value_counts().sort_index().to_string())

    print()
    print("[selectivity_max_over_mean stats]")
    print(summary["selectivity_max_over_mean"].describe().to_string())

    print()
    print("[top-10 most-fired (layer, feature)]")
    top10 = summary.nlargest(10, "n_firings_total")[
        ["layer", "feature_idx", "n_firings_total", "selectivity_max_over_mean", "max_value_observed"]
    ]
    print(top10.to_string(index=False))

    print()
    print(f"[final VRAM peak] {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GiB")
    write_torch_peak(os.environ.get("SCHEDULER_JOB_ID"))
    print(f"[done]")


if __name__ == "__main__":
    main()
