"""Phase 3 ρ=0.5 re-aggregation with extended schema for Phase 4 input.

Pre-registered design: docs/preregistrations/phase4_v001.md (§3.1, §7.2)

Re-runs Phase 3 v001.1 attribution at ρ=0.5 only; aggregates with two extensions
beyond the v001.1 schema:

  delta_m_utility_subset_U1 .. U6  (per-D_utility-subset attribution)
  delta_m_non_target_per_cat_<cat> (n_f[i] = mean(c_f[j] for j ≠ i))

Existing columns are preserved bit-identical (same forward+backward, same seed,
same RHO=0.5). A diff check vs `phase3_causal_light_pythia_160m_rho050.parquet`
verifies determinism on the original 25 columns.

Outputs:
  data/profiling/v001/phase3_causal_light_pythia_160m_rho050_v2.parquet
  configs/hash_log.json updated (1 entry: phase3_causal_light_pythia_rho050_v2_hash)

Run via scheduler:
  command:        uv run python experiments/phase3_reaggregate_pythia_160m.py
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments"))

# Re-use the validated attribution core + IO helpers from v001.1 unchanged.
from phase3_attribution_pythia_160m import (  # noqa: E402
    CATS,
    MODEL_REPO,
    MODEL_REVISION,
    PHASE2_CANDIDATES,
    SAE_REPO,
    build_layer_cand_lookup,
    compute_attribution_for_prompt,
    load_all_saes,
    load_probe_prompts,
    load_pythia,
    load_utility_prompts,
    write_torch_peak,
)

OUT_DIR = REPO_ROOT / "data" / "profiling" / "v001"
OUT_PARQUET = OUT_DIR / "phase3_causal_light_pythia_160m_rho050_v2.parquet"
ORIG_PARQUET = OUT_DIR / "phase3_causal_light_pythia_160m_rho050.parquet"
HASH_LOG = REPO_ROOT / "configs" / "hash_log.json"

RHO = 0.5
UTILITY_SUBSETS = ["U1", "U2", "U3", "U4", "U5", "U6"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def aggregate_extended(
    accum_probe: list[tuple],
    accum_utility: list[tuple],
    cand_df: pd.DataFrame,
    probe_prompts: list[dict],
    utility_prompts: list[dict],
) -> pd.DataFrame:
    """Same aggregation as v001.1 + per-cat NTE (4) + per-subset UD (6) columns."""
    qid_to_cat = {p["qid"]: p["category"] for p in probe_prompts}
    pid_to_subset = {p["prompt_id"]: p["category_id"] for p in utility_prompts}

    df_probe = pd.DataFrame(accum_probe, columns=["layer", "feature_idx", "qid", "attribution"])
    df_probe["category"] = df_probe["qid"].map(qid_to_cat)

    df_util = pd.DataFrame(accum_utility, columns=["layer", "feature_idx", "prompt_id", "attribution", "n_target_tokens"])
    df_util["ud_per_token"] = -df_util["attribution"] / df_util["n_target_tokens"].clip(lower=1)
    df_util["subset"] = df_util["prompt_id"].map(pid_to_subset)

    g_probe = df_probe.groupby(["layer", "feature_idx", "category"])["attribution"].agg(["mean", "std", "count"]).reset_index()
    g_util = df_util.groupby(["layer", "feature_idx"])["ud_per_token"].agg(["mean", "std", "count"]).reset_index()
    g_util.columns = ["layer", "feature_idx", "ud_mean", "ud_std", "ud_count"]
    g_util_subset = df_util.groupby(["layer", "feature_idx", "subset"])["ud_per_token"].mean().reset_index()
    g_util_subset.columns = ["layer", "feature_idx", "subset", "ud_mean_subset"]

    probe_idx = g_probe.set_index(["layer", "feature_idx", "category"])
    util_idx = g_util.set_index(["layer", "feature_idx"])
    util_subset_idx = g_util_subset.set_index(["layer", "feature_idx", "subset"])

    label = f"scale_down_rho{RHO}"

    rows = []
    for _, meta in cand_df.iterrows():
        L = int(meta["layer"])
        f = int(meta["feature_idx"])
        dom_cat = meta["dominant_category"]

        per_cat: dict[str, float] = {}
        for c in CATS:
            try:
                r = probe_idx.loc[(L, f, c)]
                per_cat[c] = float(r["mean"])
            except KeyError:
                per_cat[c] = float("nan")

        # per-cat NTE: n_f[i] = mean over j≠i of c_f[j]; NaN if any required c_f[j] is NaN
        nte_per_cat: dict[str, float] = {}
        for i in CATS:
            others = [per_cat[j] for j in CATS if j != i]
            if any(pd.isna(v) for v in others):
                nte_per_cat[i] = float("nan")
            else:
                nte_per_cat[i] = float(np.mean(others))

        # te_dom / nte_other (existing semantics)
        if dom_cat:
            try:
                rd = probe_idx.loc[(L, f, dom_cat)]
                te_dom = float(rd["mean"])
                te_dom_std = float(rd["std"]) if not pd.isna(rd["std"]) else float("nan")
                n_te = int(rd["count"])
            except KeyError:
                te_dom = float("nan"); te_dom_std = float("nan"); n_te = 0
            other_rows = []
            for c in CATS:
                if c == dom_cat:
                    continue
                try:
                    r = probe_idx.loc[(L, f, c)]
                    other_rows.append((float(r["mean"]), int(r["count"])))
                except KeyError:
                    pass
            if other_rows:
                means = [m for m, _ in other_rows]
                counts = [n for _, n in other_rows]
                nte_other = float(np.mean(means))
                nte_other_std = float(np.std(means, ddof=1)) if len(means) > 1 else float("nan")
                n_nte = int(sum(counts))
            else:
                nte_other = float("nan"); nte_other_std = float("nan"); n_nte = 0
        else:
            te_dom = float("nan"); te_dom_std = float("nan"); n_te = 0
            nte_other = float("nan"); nte_other_std = float("nan"); n_nte = 0

        # aggregate UD (existing) and per-subset UD (new)
        try:
            ru = util_idx.loc[(L, f)]
            ud = float(ru["ud_mean"])
            ud_std = float(ru["ud_std"]) if not pd.isna(ru["ud_std"]) else float("nan")
            n_util = int(ru["ud_count"])
        except KeyError:
            ud = float("nan"); ud_std = float("nan"); n_util = 0

        ud_per_subset: dict[str, float] = {}
        for sub in UTILITY_SUBSETS:
            try:
                r = util_subset_idx.loc[(L, f, sub)]
                ud_per_subset[sub] = float(r["ud_mean_subset"])
            except KeyError:
                ud_per_subset[sub] = float("nan")

        rows.append({
            "model_name": f"{MODEL_REPO}@{MODEL_REVISION}",
            "sae_name": SAE_REPO,
            "layer": L,
            "hook_point": "residual_post_layer",
            "feature_idx": f,
            "candidate_type": meta["candidate_type"],
            "dominant_category": dom_cat,
            "control_frac_in_dom_cat": float(meta["control_frac_in_dom_cat"]) if pd.notna(meta["control_frac_in_dom_cat"]) else float("nan"),
            "intervention_type": label,
            "intervention_rho": RHO,
            "effect_estimator": "attribution_linear",
            "delta_m_target": te_dom,
            "delta_m_target_std": te_dom_std,
            "n_prompts_target": n_te,
            "delta_m_non_target": nte_other,
            "delta_m_non_target_std": nte_other_std,
            "n_prompts_non_target": n_nte,
            "delta_m_target_per_cat_person_attribute": per_cat["person_attribute"],
            "delta_m_target_per_cat_geography": per_cat["geography"],
            "delta_m_target_per_cat_organization": per_cat["organization"],
            "delta_m_target_per_cat_occupation": per_cat["occupation"],
            "delta_m_non_target_per_cat_person_attribute": nte_per_cat["person_attribute"],
            "delta_m_non_target_per_cat_geography": nte_per_cat["geography"],
            "delta_m_non_target_per_cat_organization": nte_per_cat["organization"],
            "delta_m_non_target_per_cat_occupation": nte_per_cat["occupation"],
            "delta_m_utility": ud,
            "delta_m_utility_std": ud_std,
            "n_prompts_utility": n_util,
            "delta_m_utility_subset_U1": ud_per_subset["U1"],
            "delta_m_utility_subset_U2": ud_per_subset["U2"],
            "delta_m_utility_subset_U3": ud_per_subset["U3"],
            "delta_m_utility_subset_U4": ud_per_subset["U4"],
            "delta_m_utility_subset_U5": ud_per_subset["U5"],
            "delta_m_utility_subset_U6": ud_per_subset["U6"],
            "fallback_reason": None,
        })

    return pd.DataFrame(rows)


def verify_against_original(df_v2: pd.DataFrame) -> None:
    if not ORIG_PARQUET.exists():
        print(f"  [skip] original parquet missing: {ORIG_PARQUET.name}")
        return
    df_orig = pd.read_parquet(ORIG_PARQUET)
    assert len(df_v2) == len(df_orig), f"row mismatch: v2={len(df_v2)} orig={len(df_orig)}"
    shared_cols = [
        "layer", "feature_idx", "candidate_type", "dominant_category",
        "delta_m_target", "delta_m_non_target", "delta_m_utility",
        "delta_m_target_per_cat_person_attribute",
        "delta_m_target_per_cat_geography",
        "delta_m_target_per_cat_organization",
        "delta_m_target_per_cat_occupation",
    ]
    df_v2_sorted = df_v2.sort_values(["layer", "feature_idx"]).reset_index(drop=True)
    df_orig_sorted = df_orig.sort_values(["layer", "feature_idx"]).reset_index(drop=True)
    for c in shared_cols:
        v2_col = df_v2_sorted[c]
        orig_col = df_orig_sorted[c]
        v2_nan = pd.isna(v2_col).to_numpy()
        orig_nan = pd.isna(orig_col).to_numpy()
        nan_mismatch = int((v2_nan != orig_nan).sum())
        assert nan_mismatch == 0, f"column {c}: {nan_mismatch} NaN-pattern mismatches vs original"
        if pd.api.types.is_numeric_dtype(v2_col) and pd.api.types.is_numeric_dtype(orig_col):
            both_present = ~v2_nan
            v2 = v2_col.to_numpy()[both_present].astype(float)
            orig = orig_col.to_numpy()[both_present].astype(float)
            close = np.isclose(v2, orig, rtol=1e-5, atol=1e-7)
            n_diff = int((~close).sum())
            assert n_diff == 0, f"column {c}: {n_diff} numeric mismatches vs original"
        else:
            both_present = ~v2_nan
            v2 = v2_col.to_numpy()[both_present]
            orig = orig_col.to_numpy()[both_present]
            n_diff = int((v2 != orig).sum())
            assert n_diff == 0, f"column {c}: {n_diff} value mismatches vs original"
    print(f"  [verify] {len(shared_cols)} shared columns bit-identical to v001.1 parquet (n={len(df_v2)})")


def update_hash_log() -> None:
    log = json.loads(HASH_LOG.read_text()) if HASH_LOG.exists() else {}
    log["phase3_causal_light_pythia_rho050_v2_hash"] = {
        "file": str(OUT_PARQUET.relative_to(REPO_ROOT)),
        "sha256": sha256_of(OUT_PARQUET),
        "model": f"{MODEL_REPO}@{MODEL_REVISION}",
        "sae": SAE_REPO,
        "intervention_type": f"scale_down_rho{RHO}",
        "intervention_rho": RHO,
        "effect_estimator": "attribution_linear",
        "extends": "phase3_causal_light_pythia_rho050_hash",
        "added_columns": [
            "delta_m_non_target_per_cat_person_attribute",
            "delta_m_non_target_per_cat_geography",
            "delta_m_non_target_per_cat_organization",
            "delta_m_non_target_per_cat_occupation",
            "delta_m_utility_subset_U1",
            "delta_m_utility_subset_U2",
            "delta_m_utility_subset_U3",
            "delta_m_utility_subset_U4",
            "delta_m_utility_subset_U5",
            "delta_m_utility_subset_U6",
        ],
        "frozen_at": now_iso(),
    }
    HASH_LOG.write_text(json.dumps(log, indent=2, ensure_ascii=False, sort_keys=True))


def main() -> None:
    print(f"--- Phase 3 re-aggregation (Pythia-160m, ρ=0.5) ---")
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

    print("[load] D_probe + D_utility")
    probe_prompts = load_probe_prompts()
    utility_prompts = load_utility_prompts()
    print(f"  probe={len(probe_prompts)}  utility={len(utility_prompts)}")

    rho_sweep = [RHO]

    print("\n[stage 1] attribution loop (probe)")
    accum_probe: list[tuple] = []
    t0 = time.time()
    for i, p in enumerate(probe_prompts):
        if i % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [probe {i:4d}/{len(probe_prompts)}] {elapsed:6.1f}s")
        per_rho_per_lf, _ = compute_attribution_for_prompt(
            tok, model, saes, layer_cand_lookup, p["prompt"], p["answer"], rho_sweep)
        for (L, f), d in per_rho_per_lf[RHO].items():
            accum_probe.append((L, f, p["qid"], d))
    print(f"  probe done in {time.time()-t0:.1f}s ({len(accum_probe)} firings)")

    print("\n[stage 1b] attribution loop (utility)")
    accum_utility: list[tuple] = []
    t0 = time.time()
    for i, p in enumerate(utility_prompts):
        if i % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [utility {i:4d}/{len(utility_prompts)}] {elapsed:6.1f}s")
        per_rho_per_lf, n_tokens = compute_attribution_for_prompt(
            tok, model, saes, layer_cand_lookup, p["prompt"], p["expected_continuation"], rho_sweep)
        if n_tokens == 0:
            continue
        for (L, f), d in per_rho_per_lf[RHO].items():
            accum_utility.append((L, f, p["prompt_id"], d, n_tokens))
    print(f"  utility done in {time.time()-t0:.1f}s ({len(accum_utility)} firings)")

    print("\n[stage 2] aggregation (extended schema)")
    df = aggregate_extended(accum_probe, accum_utility, cand_df, probe_prompts, utility_prompts)
    print(f"  aggregated: {len(df)} rows × {len(df.columns)} cols")

    print("\n[stage 3] verify vs v001.1 parquet (bit-identical shared columns)")
    verify_against_original(df)

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"  saved: {OUT_PARQUET.name}")

    print("\n[stage 4] hash_log update")
    update_hash_log()
    print(f"  hash_log updated: phase3_causal_light_pythia_rho050_v2_hash")

    write_torch_peak(os.environ.get("SCHEDULER_JOB_ID"))
    print(f"\n[final VRAM] {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GiB")
    print("[done] Phase 4 cluster script の入力として使える")


if __name__ == "__main__":
    main()
