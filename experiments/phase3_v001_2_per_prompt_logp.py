"""Phase 3 v001.2 — per-prompt logP retention (Pythia-160m, ρ=0.5).

Pre-registered design: docs/preregistrations/phase4_v001_1_measurement.md §3.3

Re-runs Phase 3 ρ=0.5 attribution + captures per-(layer, feature, prompt_id):
  logP_baseline_target   : full target log-prob from baseline forward
  logP_intervened_target : baseline + linear ΔlogP under scale_down ρ=0.5
  delta_logP_attribution : (ρ-1) · Σ val·grad_dot, summed over firing positions

Long-format output enables prompt-level bootstrap CI without re-forward.

Outputs:
  data/profiling/v001/phase3_causal_light_pythia_160m_rho050_v3.parquet
  configs/hash_log.json updated (1 entry: phase3_causal_light_pythia_rho050_v3_hash)

Verification: groupby (layer, feature_idx, category).mean(delta_logP_attribution)
on probe rows reproduces v2 parquet's delta_m_target_per_cat_* bit-identically.

Run via scheduler:
  command:        uv run python experiments/phase3_v001_2_per_prompt_logp.py
  requested_gpus: [0]
  vram_budget_gib: 16
  env: { SCHEDULER_JOB_ID: "<job_id>" }
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from phase3_attribution_pythia_160m import (  # noqa: E402
    CATS,
    MAX_TOKENS,
    MODEL_REPO,
    MODEL_REVISION,
    PHASE2_CANDIDATES,
    SAE_REPO,
    build_layer_cand_lookup,
    load_all_saes,
    load_probe_prompts,
    load_pythia,
    load_utility_prompts,
    write_torch_peak,
)

DEVICE = "cuda:0"
RHO = 0.5
UTILITY_SUBSETS = ["U1", "U2", "U3", "U4", "U5", "U6"]

OUT_DIR = REPO_ROOT / "data" / "profiling" / "v001"
OUT_PARQUET = OUT_DIR / "phase3_causal_light_pythia_160m_rho050_v3.parquet"
V2_PARQUET = OUT_DIR / "phase3_causal_light_pythia_160m_rho050_v2.parquet"
HASH_LOG = REPO_ROOT / "configs" / "hash_log.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_attribution_with_baseline(
    tok,
    model,
    saes: list[dict],
    layer_cand_lookup: list[torch.Tensor],
    prompt_text: str,
    answer_text: str,
    rho: float,
) -> tuple[dict[tuple[int, int], float], float, int]:
    """Single-ρ variant of compute_attribution_for_prompt that also returns
    baseline target log-prob.

    Returns (per_lf_delta, baseline_logP_target, n_answer_tokens).
    per_lf_delta[(layer, feature_idx)] = (ρ-1) · Σ val·grad_dot over firings.

    Forward+backward identical to phase3_attribution_pythia_160m.compute_*; this
    function differs only in (a) single ρ and (b) returning baseline_logP.
    """
    full_text = prompt_text.rstrip() + " " + answer_text.lstrip()
    enc_full = tok(full_text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS).to(DEVICE)
    enc_prompt = tok(prompt_text.rstrip(), return_tensors="pt", truncation=True, max_length=MAX_TOKENS).to(DEVICE)

    full_ids = enc_full.input_ids[0]
    n_total = full_ids.shape[0]
    n_prompt = enc_prompt.input_ids.shape[1]
    if n_prompt >= n_total:
        return {}, float("nan"), 0
    n_answer = n_total - n_prompt
    answer_ids = full_ids[n_prompt:n_total]
    pred_positions = torch.arange(n_prompt - 1, n_total - 1, device=DEVICE)

    model.zero_grad()
    outputs = model(
        input_ids=full_ids.unsqueeze(0),
        output_hidden_states=True,
        use_cache=False,
    )
    n_layers = len(saes)
    for L in range(n_layers):
        outputs.hidden_states[L + 1].retain_grad()

    logits = outputs.logits[0]
    log_probs = F.log_softmax(logits[pred_positions], dim=-1)
    target_log_p = log_probs.gather(1, answer_ids.unsqueeze(-1)).sum()
    baseline_logP = float(target_log_p.item())
    target_log_p.backward()

    per_lf: dict[tuple[int, int], float] = {}
    factor = rho - 1.0

    for L in range(n_layers):
        h = outputs.hidden_states[L + 1][0].detach()
        g = outputs.hidden_states[L + 1].grad[0].detach()
        sae = saes[L]
        weights = sae["weights"]
        cfg = sae["cfg"]

        with torch.no_grad():
            sae_in = h - weights["b_dec"]
            pre_acts = sae_in @ weights["encoder.weight"].T + weights["encoder.bias"]
            topk = pre_acts.topk(int(cfg["k"]), dim=-1)
            vals = topk.values
            if not bool(cfg.get("signed", False)):
                vals = vals.relu()
            idxs = topk.indices

            cand_mask = layer_cand_lookup[L][idxs]
            active_mask = cand_mask & (vals > 0)
            if not active_mask.any():
                continue

            active_pos, active_k = active_mask.nonzero(as_tuple=True)
            active_idxs = idxs[active_pos, active_k]
            active_vals = vals[active_pos, active_k]
            active_W = weights["W_dec"][active_idxs]
            active_g = g[active_pos]
            grad_dot = (active_W * active_g).sum(-1)
            base_units = (active_vals * grad_dot).tolist()
            active_idxs_list = active_idxs.tolist()

            for f, base in zip(active_idxs_list, base_units):
                key = (L, f)
                per_lf[key] = per_lf.get(key, 0.0) + factor * base

    return per_lf, baseline_logP, n_answer


def verify_against_v2(df_v3: pd.DataFrame) -> None:
    """Aggregate v3 long-format probe rows by (layer, feature_idx, category) and
    confirm bit-identical match against v2 parquet's delta_m_target_per_cat_*
    columns (same forward implies same numerics)."""
    if not V2_PARQUET.exists():
        print(f"  [skip] v2 parquet missing: {V2_PARQUET.name}")
        return
    df_v2 = pd.read_parquet(V2_PARQUET)
    v3_probe = df_v3[df_v3["source"] == "probe"].copy()
    v3_agg = (
        v3_probe.groupby(["layer", "feature_idx", "category"])["delta_logP_attribution"]
        .mean()
        .reset_index()
    )

    n_check = 0
    n_diff = 0
    for cat in CATS:
        col_v2 = f"delta_m_target_per_cat_{cat}"
        sub = v3_agg[v3_agg["category"] == cat][["layer", "feature_idx", "delta_logP_attribution"]]
        sub = sub.rename(columns={"delta_logP_attribution": "v3_mean"})
        merged = df_v2[["layer", "feature_idx", col_v2]].merge(
            sub, on=["layer", "feature_idx"], how="left"
        )
        v2_vals = merged[col_v2].to_numpy()
        v3_vals = merged["v3_mean"].to_numpy()
        v2_nan = pd.isna(v2_vals)
        v3_nan = pd.isna(v3_vals)
        nan_mismatch = int((v2_nan != v3_nan).sum())
        assert nan_mismatch == 0, f"{cat}: {nan_mismatch} NaN-pattern mismatches v2 vs v3-agg"
        both_present = ~v2_nan
        v2_arr = v2_vals[both_present].astype(float)
        v3_arr = v3_vals[both_present].astype(float)
        diff = int((~np.isclose(v2_arr, v3_arr, rtol=1e-5, atol=1e-7)).sum())
        n_check += int(both_present.sum())
        n_diff += diff
    assert n_diff == 0, f"v3 aggregate vs v2: {n_diff} numeric mismatches across {n_check} cells"
    print(f"  [verify] v3 long-format aggregates bit-identically to v2 per-cat columns ({n_check} cells)")


def update_hash_log() -> None:
    log = json.loads(HASH_LOG.read_text()) if HASH_LOG.exists() else {}
    log["phase3_causal_light_pythia_rho050_v3_hash"] = {
        "file": str(OUT_PARQUET.relative_to(REPO_ROOT)),
        "sha256": sha256_of(OUT_PARQUET),
        "model": f"{MODEL_REPO}@{MODEL_REVISION}",
        "sae": SAE_REPO,
        "intervention_type": f"scale_down_rho{RHO}",
        "intervention_rho": RHO,
        "effect_estimator": "attribution_linear",
        "extends": "phase3_causal_light_pythia_rho050_v2_hash",
        "schema": "long_format_per_prompt",
        "row_keys": ["layer", "feature_idx", "prompt_id"],
        "added_columns": [
            "logP_baseline_target",
            "logP_intervened_target",
            "delta_logP_attribution",
            "source",
            "category",
            "split_role",
            "n_target_tokens",
        ],
        "frozen_at": now_iso(),
    }
    HASH_LOG.write_text(json.dumps(log, indent=2, ensure_ascii=False, sort_keys=True))


def main() -> None:
    print(f"--- Phase 3 v001.2 per-prompt logP (Pythia-160m, ρ={RHO}) ---")
    print(f"  device:    {torch.cuda.get_device_name(0)}")
    print(f"  torch:     {torch.__version__}")
    print(f"  output:    {OUT_PARQUET.name}")

    print("[load] Pythia + 12 SAEs")
    tok, model = load_pythia()
    n_layers = model.config.num_hidden_layers
    saes = load_all_saes(n_layers)
    num_latents = saes[0]["weights"]["encoder.weight"].shape[0]
    print(f"  n_layers={n_layers}, num_latents={num_latents}")
    print(f"  param VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GiB")

    print("[load] Phase 2 candidates (cs + broad)")
    cand_df = pd.read_parquet(PHASE2_CANDIDATES)
    cand_df = cand_df[cand_df["candidate_type"].isin(["category_selective", "broad"])].reset_index(drop=True)
    print(f"  candidates: {len(cand_df)}")
    layer_cand_lookup = build_layer_cand_lookup(cand_df, n_layers, num_latents)

    print("[load] D_probe positive+paraphrase")
    probe_prompts = load_probe_prompts()
    qid_to_cat = {p["qid"]: p["category"] for p in probe_prompts}
    qid_to_split = {p["qid"]: p["split_role"] for p in probe_prompts}
    print(f"  probe={len(probe_prompts)}")

    print("[load] D_utility")
    utility_prompts = load_utility_prompts()
    pid_to_subset = {p["prompt_id"]: p["category_id"] for p in utility_prompts}
    print(f"  utility={len(utility_prompts)}")

    rows: list[tuple] = []  # (layer, feature_idx, prompt_id, source, category, split_role, baseline, intervened, delta, n_target_tokens)

    print("\n[stage 1] attribution loop (probe)")
    t0 = time.time()
    for i, p in enumerate(probe_prompts):
        if i % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [probe {i:4d}/{len(probe_prompts)}] {elapsed:6.1f}s")
        per_lf, baseline_logP, n_answer = compute_attribution_with_baseline(
            tok, model, saes, layer_cand_lookup, p["prompt"], p["answer"], RHO
        )
        if n_answer == 0:
            continue
        cat = qid_to_cat[p["qid"]]
        split_role = qid_to_split[p["qid"]]
        for (L, f), delta in per_lf.items():
            rows.append((
                int(L), int(f), p["qid"], "probe", cat, split_role,
                float(baseline_logP), float(baseline_logP + delta), float(delta), int(n_answer),
            ))
    print(f"  probe done in {time.time()-t0:.1f}s ({len(rows)} firings)")

    n_after_probe = len(rows)

    print("\n[stage 1b] attribution loop (utility)")
    t0 = time.time()
    for i, p in enumerate(utility_prompts):
        if i % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [utility {i:4d}/{len(utility_prompts)}] {elapsed:6.1f}s")
        per_lf, baseline_logP, n_answer = compute_attribution_with_baseline(
            tok, model, saes, layer_cand_lookup, p["prompt"], p["expected_continuation"], RHO
        )
        if n_answer == 0:
            continue
        subset = pid_to_subset[p["prompt_id"]]
        for (L, f), delta in per_lf.items():
            rows.append((
                int(L), int(f), p["prompt_id"], "utility", subset, "n/a",
                float(baseline_logP), float(baseline_logP + delta), float(delta), int(n_answer),
            ))
    print(f"  utility done in {time.time()-t0:.1f}s ({len(rows) - n_after_probe} firings)")

    print(f"\n[stage 2] build long-format DataFrame ({len(rows)} rows)")
    df = pd.DataFrame(rows, columns=[
        "layer", "feature_idx", "prompt_id", "source", "category", "split_role",
        "logP_baseline_target", "logP_intervened_target", "delta_logP_attribution", "n_target_tokens",
    ])
    df = df.astype({
        "layer": "int32",
        "feature_idx": "int32",
        "logP_baseline_target": "float32",
        "logP_intervened_target": "float32",
        "delta_logP_attribution": "float32",
        "n_target_tokens": "int32",
    })
    print(f"  schema: {dict(df.dtypes)}")
    print(f"  source distribution: {df['source'].value_counts().to_dict()}")
    print(f"  split_role distribution (probe): {df[df.source == 'probe']['split_role'].value_counts().to_dict()}")

    print("\n[stage 3] verify aggregation against v2 parquet (bit-identical per-cat)")
    verify_against_v2(df)

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"  saved: {OUT_PARQUET.name} ({OUT_PARQUET.stat().st_size / 1024**2:.1f} MiB)")

    print("\n[stage 4] hash_log update")
    update_hash_log()
    print(f"  hash_log updated: phase3_causal_light_pythia_rho050_v3_hash")

    write_torch_peak(os.environ.get("SCHEDULER_JOB_ID"))
    print(f"\n[final VRAM] {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GiB")
    print("[done] Phase 4 v001.1 stratify script の入力として使える")


if __name__ == "__main__":
    main()
