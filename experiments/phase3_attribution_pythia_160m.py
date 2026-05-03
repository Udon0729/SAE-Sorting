"""Phase 3 light causal — attribution patching for Pythia-160m-deduped.

Reference: docs/dataset.md §8 Phase 3 + docs/implementation.md §8.3 +
docs/project_attribution_patching_protocol.md.

For each candidate (layer, feature) from Phase 2 (cs + broad = 10,311 entries),
compute attribution patching estimates of:

  TargetEffect (TE_dom)        : mean ΔlogP(answer) over D_probe positive+paraphrase
                                 prompts in the feature's dominant_category
  NonTargetEffect (NTE_other)  : mean ΔlogP(answer) over D_probe positive+paraphrase
                                 prompts NOT in dominant_category (3 other cats)
  UtilityDamage (UD)           : mean -ΔlogP(continuation) / n_target_tokens over
                                 D_utility prompts (perplexity log proxy)

Intervention semantics (linear approximation, scale-down ρ=0.5):
  z'_t = z_t + (ρ - 1) · val_f_t · W_dec[f]    where ρ = 0.5
  ΔM(f, t, p) ≈ (ρ - 1) · val_f_t · (W_dec[f] · ∇M(z_t))
              = -0.5 · val_f_t · (W_dec[f] · ∇M(z_t))

Why scale-down ρ=0.5 instead of mean replacement: implementation.md §8.3 lists
both as Phase 3 options. For Top-K SAE features, "mean replacement" with the
firing-conditional mean (~ feature's typical firing magnitude) gives
(mean - val) ≈ 0 on dominant-category prompts (where val ≈ mean), producing
near-zero attribution; an unconditional-mean baseline collapses to ~0 for sparse
TopK features and is functionally equivalent to zero ablation. ρ=0.5 instead
provides a well-defined non-zero perturbation, matches a documented option,
and yields good linear approximation (smaller perturbation than zero ablation).

Validation: random 200 (layer, feature, prompt) pairs vs real intervention
(forward hook scale-down at firing positions); acceptance gate Pearson r ≥ 0.85,
Spearman ρ ≥ 0.80, filtered rel_err mean ≤ 0.20 (rel_err computed only on pairs
with |real_d| > REL_ERR_FLOOR to avoid near-zero division noise).

v001 scope:
  - Pythia-160m-deduped only (Qwen Phase 3 in v002)
  - scale_down ρ=0.5 intervention only (mean replacement in v002 if applicable)
  - 200 random (feature, prompt) pair validation (full 9,600 in v002)
  - rare/mixed Phase-2 features ignored (focus on cs+broad candidates)

Outputs:
  data/profiling/v001/phase3_causal_light_pythia_160m.parquet
  data/profiling/v001/phase3_validation_pythia_160m.json
  configs/hash_log.json updated (2 new entries)

Run via scheduler (preferred):
  command:        uv run python experiments/phase3_attribution_pythia_160m.py
  requested_gpus: [0]
  vram_budget_gib: 16
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
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer, GPTNeoXForCausalLM

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE_PROMPTS = REPO_ROOT / "data" / "probe" / "v001" / "prompts.jsonl"
UTILITY_PROMPTS = REPO_ROOT / "data" / "utility" / "v001" / "prompts.jsonl"
PHASE1_FEATURES = REPO_ROOT / "data" / "profiling" / "v001" / "phase1_features.parquet"
PHASE2_CANDIDATES = REPO_ROOT / "data" / "profiling" / "v001" / "phase2_candidates_pythia_160m.parquet"
OUT_PARQUET = REPO_ROOT / "data" / "profiling" / "v001" / "phase3_causal_light_pythia_160m.parquet"
OUT_VALIDATION = REPO_ROOT / "data" / "profiling" / "v001" / "phase3_validation_pythia_160m.json"
HASH_LOG = REPO_ROOT / "configs" / "hash_log.json"

MODEL_REPO = "EleutherAI/pythia-160m-deduped"
MODEL_REVISION = "step143000"
SAE_REPO = "EleutherAI/sae-pythia-160m-deduped-32k"
DEVICE = "cuda:0"
DTYPE = torch.float32
MAX_TOKENS = 128  # prompt + answer combined
N_VALIDATION_PAIRS = 200
VALIDATION_SEED = 42

INTERVENTION_RHO = 0.5  # scale_down: val_new = ρ · val
INTERVENTION_TYPE_LABEL = f"scale_down_rho{INTERVENTION_RHO}"

ACCEPT_PEARSON = 0.85
ACCEPT_SPEARMAN = 0.80
ACCEPT_REL_ERR = 0.20
REL_ERR_FLOOR = 1e-3  # |real_d| < FLOOR pairs excluded from rel_err mean (numerator/denominator both ≈ 0)

CATS = ["person_attribute", "geography", "organization", "occupation"]
PROBE_TYPES = {"positive", "paraphrase"}


# -- helpers ------------------------------------------------------------------

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


def build_mean_lookup(phase1_df: pd.DataFrame, n_layers: int, num_latents: int) -> list[torch.Tensor]:
    """Per-layer (num_latents,) tensor of mean_value from Phase 1; 0 for non-firing features.

    Retained for v002 mean-replacement intervention; not used in v001 scale_down path.
    """
    lookup: list[torch.Tensor] = []
    for L in range(n_layers):
        sub = phase1_df[phase1_df["layer"] == L]
        t = torch.zeros(num_latents, dtype=DTYPE, device=DEVICE)
        if len(sub):
            idxs = torch.tensor(sub["feature_idx"].to_numpy(), dtype=torch.long, device=DEVICE)
            vals = torch.tensor(sub["mean_value"].to_numpy(), dtype=DTYPE, device=DEVICE)
            t[idxs] = vals
        lookup.append(t)
    return lookup


def build_layer_cand_lookup(cand_df: pd.DataFrame, n_layers: int, num_latents: int) -> list[torch.Tensor]:
    """Per-layer (num_latents,) bool mask of candidate features (cs + broad only)."""
    lookup: list[torch.Tensor] = []
    for L in range(n_layers):
        sub = cand_df[cand_df["layer"] == L]
        mask = torch.zeros(num_latents, dtype=torch.bool, device=DEVICE)
        if len(sub):
            idxs = torch.tensor(sub["feature_idx"].to_numpy(), dtype=torch.long, device=DEVICE)
            mask[idxs] = True
        lookup.append(mask)
    return lookup


def load_probe_prompts() -> list[dict]:
    rows = []
    with PROBE_PROMPTS.open("r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p["prompt_type"] in PROBE_TYPES:
                rows.append(p)
    return rows


def load_utility_prompts() -> list[dict]:
    rows = []
    with UTILITY_PROMPTS.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


# -- attribution core ---------------------------------------------------------

def compute_attribution_for_prompt(
    tok,
    model,
    saes: list[dict],
    mean_lookup: list[torch.Tensor],  # unused in v001 scale_down path; kept for v002
    layer_cand_lookup: list[torch.Tensor],
    prompt_text: str,
    answer_text: str,
) -> tuple[dict[tuple[int, int], float], int]:
    """Returns (per_lf, n_answer_tokens).
    per_lf maps (layer, feature_idx) -> sum over firing positions of ΔlogP linear estimate
    under scale_down ρ=INTERVENTION_RHO intervention.
    Only candidate features (cs + broad) are tracked.
    """
    full_text = prompt_text.rstrip() + " " + answer_text.lstrip()
    enc_full = tok(full_text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS).to(DEVICE)
    enc_prompt = tok(prompt_text.rstrip(), return_tensors="pt", truncation=True, max_length=MAX_TOKENS).to(DEVICE)

    full_ids = enc_full.input_ids[0]
    n_total = full_ids.shape[0]
    n_prompt = enc_prompt.input_ids.shape[1]
    if n_prompt >= n_total:
        return {}, 0
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
    target_log_p.backward()

    per_lf: dict[tuple[int, int], float] = {}
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
            # scale_down intervention: val_new = ρ · val ⇒ Δval = (ρ - 1) · val
            delta = (INTERVENTION_RHO - 1.0) * active_vals * grad_dot

            # Sum per feature (a feature may fire at multiple positions for this prompt)
            for f, d in zip(active_idxs.tolist(), delta.tolist()):
                key = (L, f)
                per_lf[key] = per_lf.get(key, 0.0) + d

    return per_lf, n_answer


# -- real intervention (validation) ------------------------------------------

def real_intervention_forward(
    tok,
    model,
    sae: dict,
    layer: int,
    feature: int,
    rho: float,
    prompt_text: str,
    answer_text: str,
) -> float | None:
    """Returns ΔlogP = logP_intervened - logP_baseline (full-answer log-prob), or None if truncated.
    Real intervention: scale val_new = ρ · val at firing positions of `feature` at `layer`.
    """
    full_text = prompt_text.rstrip() + " " + answer_text.lstrip()
    enc_full = tok(full_text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS).to(DEVICE)
    enc_prompt = tok(prompt_text.rstrip(), return_tensors="pt", truncation=True, max_length=MAX_TOKENS).to(DEVICE)
    full_ids = enc_full.input_ids
    n_total = full_ids.shape[1]
    n_prompt = enc_prompt.input_ids.shape[1]
    if n_prompt >= n_total:
        return None
    answer_ids = full_ids[0, n_prompt:n_total]
    pred_positions = torch.arange(n_prompt - 1, n_total - 1, device=DEVICE)

    weights = sae["weights"]
    cfg = sae["cfg"]
    K = int(cfg["k"])

    def target_logp(out) -> float:
        return F.log_softmax(out.logits[0, pred_positions], dim=-1).gather(
            1, answer_ids.unsqueeze(-1)).sum().item()

    with torch.no_grad():
        baseline_out = model(full_ids, use_cache=False)
        baseline_logp = target_logp(baseline_out)

    def intervention_hook(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        sae_in = h[0] - weights["b_dec"]
        pre = sae_in @ weights["encoder.weight"].T + weights["encoder.bias"]
        topk = pre.topk(K, dim=-1)
        vals = topk.values
        if not bool(cfg.get("signed", False)):
            vals = vals.relu()
        idxs = topk.indices
        match = (idxs == feature)
        firing_mask = match.any(dim=-1)
        if not firing_mask.any():
            return output
        firing_pos = firing_mask.nonzero(as_tuple=True)[0]
        firing_k_idx = match[firing_pos].float().argmax(dim=-1)
        firing_vals = vals[firing_pos, firing_k_idx]
        # scale_down: Δresidual = (ρ - 1) · val · W_dec[f] at firing positions
        delta = ((rho - 1.0) * firing_vals).unsqueeze(-1) * weights["W_dec"][feature].unsqueeze(0)
        h_new = h.clone()
        h_new[0, firing_pos] = h_new[0, firing_pos] + delta
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new

    handle = model.gpt_neox.layers[layer].register_forward_hook(intervention_hook)
    try:
        with torch.no_grad():
            int_out = model(full_ids, use_cache=False)
            int_logp = target_logp(int_out)
    finally:
        handle.remove()

    return int_logp - baseline_logp


def run_validation(
    tok,
    model,
    saes: list[dict],
    mean_lookup: list[torch.Tensor],
    accum_probe: list[tuple],
    accum_utility: list[tuple],
    probe_prompts: list[dict],
    utility_prompts: list[dict],
    n_pairs: int,
) -> dict:
    rng = np.random.default_rng(VALIDATION_SEED)
    qid_to_probe = {p["qid"]: p for p in probe_prompts}
    pid_to_utility = {p["prompt_id"]: p for p in utility_prompts}

    samples: list[tuple] = []
    for L, f, qid, d in accum_probe:
        p = qid_to_probe[qid]
        samples.append((L, f, p["prompt"], p["answer"], d, "probe", qid))
    for L, f, pid, d, _ in accum_utility:
        p = pid_to_utility[pid]
        samples.append((L, f, p["prompt"], p["expected_continuation"], d, "utility", pid))

    if len(samples) < n_pairs:
        n_pairs = len(samples)
    sample_indices = rng.choice(len(samples), size=n_pairs, replace=False)

    real_deltas: list[float] = []
    linear_deltas: list[float] = []
    failed_pairs: list[dict] = []
    skipped = 0

    t0 = time.time()
    for i, idx in enumerate(sample_indices):
        if i % 25 == 0:
            print(f"  [validation {i}/{n_pairs}] {time.time()-t0:.1f}s")
        L, f, prompt_text, answer_text, linear_d, source, item_id = samples[idx]
        real_d = real_intervention_forward(
            tok, model, saes[L], L, f, INTERVENTION_RHO, prompt_text, answer_text)
        if real_d is None:
            skipped += 1
            continue
        real_deltas.append(real_d)
        linear_deltas.append(linear_d)
        if abs(real_d) > REL_ERR_FLOOR:
            rel = abs(real_d - linear_d) / abs(real_d)
            if rel > 0.5:
                failed_pairs.append({
                    "layer": int(L), "feature_idx": int(f), "item_id": item_id, "source": source,
                    "delta_real": float(real_d), "delta_linear": float(linear_d), "rel_err": float(rel),
                })

    real_arr = np.array(real_deltas)
    linear_arr = np.array(linear_deltas)
    if len(real_arr) > 2:
        pearson_r = float(pearsonr(real_arr, linear_arr).statistic)
        spearman_rho = float(spearmanr(real_arr, linear_arr).statistic)
    else:
        pearson_r = float("nan")
        spearman_rho = float("nan")

    # Filtered rel_err: compute only on pairs with non-trivial real effect
    nonzero_mask = np.abs(real_arr) > REL_ERR_FLOOR
    n_nonzero = int(nonzero_mask.sum())
    if n_nonzero > 0:
        rel_errs_nz = np.abs(real_arr[nonzero_mask] - linear_arr[nonzero_mask]) / np.abs(real_arr[nonzero_mask])
        rel_err_mean_nz = float(rel_errs_nz.mean())
        rel_err_p50_nz = float(np.percentile(rel_errs_nz, 50))
        rel_err_p95_nz = float(np.percentile(rel_errs_nz, 95))
    else:
        rel_err_mean_nz = float("nan")
        rel_err_p50_nz = float("nan")
        rel_err_p95_nz = float("nan")

    # Global relative MAE (alternative robust metric)
    if real_arr.size > 0:
        mean_abs_real = float(np.mean(np.abs(real_arr)))
        rel_mae = float(np.mean(np.abs(real_arr - linear_arr))) / max(mean_abs_real, 1e-12)
    else:
        rel_mae = float("nan")

    accept = (
        pearson_r >= ACCEPT_PEARSON
        and spearman_rho >= ACCEPT_SPEARMAN
        and rel_err_mean_nz == rel_err_mean_nz  # not NaN
        and rel_err_mean_nz <= ACCEPT_REL_ERR
    )

    return {
        "intervention_type": INTERVENTION_TYPE_LABEL,
        "intervention_rho": INTERVENTION_RHO,
        "n_validation_pairs": int(len(real_deltas)),
        "n_skipped": int(skipped),
        "n_nontrivial_pairs": n_nonzero,
        "rel_err_floor": REL_ERR_FLOOR,
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "rel_err_mean_filtered": rel_err_mean_nz,
        "rel_err_p50_filtered": rel_err_p50_nz,
        "rel_err_p95_filtered": rel_err_p95_nz,
        "rel_mae_global": rel_mae,
        "accept": bool(accept),
        "acceptance_thresholds": {
            "pearson_r_min": ACCEPT_PEARSON,
            "spearman_rho_min": ACCEPT_SPEARMAN,
            "rel_err_mean_filtered_max": ACCEPT_REL_ERR,
        },
        "failed_pairs_high_rel_err": failed_pairs[:50],
        "frozen_at": now_iso(),
    }


# -- aggregation --------------------------------------------------------------

def aggregate_per_feature(
    accum_probe: list[tuple],
    accum_utility: list[tuple],
    cand_df: pd.DataFrame,
    probe_prompts: list[dict],
    accept_validation: bool,
) -> pd.DataFrame:
    qid_to_cat = {p["qid"]: p["category"] for p in probe_prompts}

    df_probe = pd.DataFrame(accum_probe, columns=["layer", "feature_idx", "qid", "attribution"])
    df_probe["category"] = df_probe["qid"].map(qid_to_cat)

    df_util = pd.DataFrame(accum_utility, columns=["layer", "feature_idx", "prompt_id", "attribution", "n_target_tokens"])
    df_util["ud_per_token"] = -df_util["attribution"] / df_util["n_target_tokens"].clip(lower=1)

    # Pre-aggregate probe per (L, f, cat)
    g_probe = df_probe.groupby(["layer", "feature_idx", "category"])["attribution"].agg(["mean", "std", "count"]).reset_index()
    # Pre-aggregate utility per (L, f)
    g_util = df_util.groupby(["layer", "feature_idx"])["ud_per_token"].agg(["mean", "std", "count"]).reset_index()
    g_util.columns = ["layer", "feature_idx", "ud_mean", "ud_std", "ud_count"]

    # Build fast lookup
    probe_idx = g_probe.set_index(["layer", "feature_idx", "category"])
    util_idx = g_util.set_index(["layer", "feature_idx"])

    fallback = None if accept_validation else "attribution_low_correlation"

    rows = []
    for _, meta in cand_df.iterrows():
        L = int(meta["layer"])
        f = int(meta["feature_idx"])
        dom_cat = meta["dominant_category"]

        per_cat = {}
        for c in CATS:
            try:
                r = probe_idx.loc[(L, f, c)]
                per_cat[c] = float(r["mean"])
            except KeyError:
                per_cat[c] = float("nan")

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
                # Mean of per-cat means (NTE_other defined as average over other 3 cats)
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

        try:
            ru = util_idx.loc[(L, f)]
            ud = float(ru["ud_mean"])
            ud_std = float(ru["ud_std"]) if not pd.isna(ru["ud_std"]) else float("nan")
            n_util = int(ru["ud_count"])
        except KeyError:
            ud = float("nan"); ud_std = float("nan"); n_util = 0

        rows.append({
            "model_name": f"{MODEL_REPO}@{MODEL_REVISION}",
            "sae_name": SAE_REPO,
            "layer": L,
            "hook_point": "residual_post_layer",
            "feature_idx": f,
            "candidate_type": meta["candidate_type"],
            "dominant_category": dom_cat,
            "control_frac_in_dom_cat": float(meta["control_frac_in_dom_cat"]) if pd.notna(meta["control_frac_in_dom_cat"]) else float("nan"),
            "intervention_type": INTERVENTION_TYPE_LABEL,
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
            "delta_m_utility": ud,
            "delta_m_utility_std": ud_std,
            "n_prompts_utility": n_util,
            "fallback_reason": fallback,
        })

    return pd.DataFrame(rows)


# -- output -------------------------------------------------------------------

def update_hash_log(df_aggregate: pd.DataFrame, validation: dict) -> None:
    log = json.loads(HASH_LOG.read_text()) if HASH_LOG.exists() else {}
    log["phase3_causal_light_pythia_hash"] = {
        "file": str(OUT_PARQUET.relative_to(REPO_ROOT)),
        "sha256": sha256_of(OUT_PARQUET),
        "n_candidates": int(len(df_aggregate)),
        "model": f"{MODEL_REPO}@{MODEL_REVISION}",
        "sae": SAE_REPO,
        "intervention_type": INTERVENTION_TYPE_LABEL,
        "effect_estimator": "attribution_linear",
        "frozen_at": now_iso(),
    }
    log["phase3_validation_pythia_hash"] = {
        "file": str(OUT_VALIDATION.relative_to(REPO_ROOT)),
        "sha256": sha256_of(OUT_VALIDATION),
        "accept": bool(validation["accept"]),
        "pearson_r": validation["pearson_r"],
        "spearman_rho": validation["spearman_rho"],
        "rel_err_mean_filtered": validation["rel_err_mean_filtered"],
        "rel_mae_global": validation["rel_mae_global"],
        "n_nontrivial_pairs": validation["n_nontrivial_pairs"],
        "frozen_at": now_iso(),
    }
    HASH_LOG.write_text(json.dumps(log, indent=2, ensure_ascii=False, sort_keys=True))


def print_sanity(df: pd.DataFrame) -> None:
    cs = df[df["candidate_type"] == "category_selective"]
    valid = cs.dropna(subset=["delta_m_target", "delta_m_non_target"])
    if len(valid):
        ratio = float((valid["delta_m_target"].abs() > valid["delta_m_non_target"].abs()).mean())
        print(f"  cs candidates: |TE_dom| > |NTE_other| ratio: {ratio:.2%} (target ≥ 70%)")
        print(f"  cs candidates: mean(TE_dom)        = {valid['delta_m_target'].mean():.4f}")
        print(f"  cs candidates: mean(NTE_other)     = {valid['delta_m_non_target'].mean():.4f}")
        print(f"  cs candidates: mean(|TE_dom|)      = {valid['delta_m_target'].abs().mean():.4f}")
        print(f"  cs candidates: mean(|NTE_other|)   = {valid['delta_m_non_target'].abs().mean():.4f}")
    print()
    print("  Per-layer mean (cs+broad):")
    for L in sorted(df["layer"].unique()):
        sub = df[df["layer"] == L]
        te_mean = sub["delta_m_target"].mean()
        nte_mean = sub["delta_m_non_target"].mean()
        ud_mean = sub["delta_m_utility"].mean()
        print(f"  L{int(L):2d}: TE={te_mean: .4f}  NTE={nte_mean: .4f}  UD={ud_mean: .4f}  n={len(sub)}")


def write_torch_peak(job_id: str | None) -> None:
    if not job_id:
        return
    target = REPO_ROOT / "ops" / "queue" / "logs" / f"{job_id}_torch_peak.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"0": int(torch.cuda.max_memory_allocated(0))}))


# -- main ---------------------------------------------------------------------

def main() -> None:
    print(f"--- Phase 3 light causal (attribution patching) — Pythia-160m ---")
    print(f"  device: {torch.cuda.get_device_name(0)}")
    print(f"  torch:  {torch.__version__}")

    print("[load] Pythia + 12 SAEs")
    tok, model = load_pythia()
    n_layers = model.config.num_hidden_layers
    saes = load_all_saes(n_layers)
    num_latents = saes[0]["weights"]["encoder.weight"].shape[0]
    print(f"  n_layers={n_layers}, num_latents={num_latents}")
    print(f"  param VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GiB")

    print("[load] Phase 1 features (mean_value lookup)")
    phase1_df = pd.read_parquet(PHASE1_FEATURES)
    mean_lookup = build_mean_lookup(phase1_df, n_layers, num_latents)
    print(f"  phase1 rows: {len(phase1_df)}")

    print("[load] Phase 2 candidates (cs + broad)")
    cand_df = pd.read_parquet(PHASE2_CANDIDATES)
    cand_df = cand_df[cand_df["candidate_type"].isin(["category_selective", "broad"])].reset_index(drop=True)
    print(f"  candidates: {len(cand_df)}")
    layer_cand_lookup = build_layer_cand_lookup(cand_df, n_layers, num_latents)
    cand_per_layer = cand_df.groupby("layer").size().to_dict()
    print(f"  per-layer counts: {dict(sorted(cand_per_layer.items()))}")

    print("[load] D_probe positive+paraphrase")
    probe_prompts = load_probe_prompts()
    print(f"  probe: {len(probe_prompts)}")

    print("[load] D_utility")
    utility_prompts = load_utility_prompts()
    print(f"  utility: {len(utility_prompts)}")

    # === Stage 1: attribution loop ===
    print("\n[stage 1] attribution loop (probe)")
    accum_probe: list[tuple] = []
    t0 = time.time()
    for i, p in enumerate(probe_prompts):
        if i % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            print(f"  [probe {i:4d}/{len(probe_prompts)}] {elapsed:6.1f}s ({rate:.1f}/s)")
        per_lf, _ = compute_attribution_for_prompt(
            tok, model, saes, mean_lookup, layer_cand_lookup, p["prompt"], p["answer"])
        for (L, f), d in per_lf.items():
            accum_probe.append((L, f, p["qid"], d))
    print(f"  probe done: {len(accum_probe)} firings in {time.time()-t0:.1f}s")

    print("\n[stage 1b] attribution loop (utility)")
    accum_utility: list[tuple] = []
    t0 = time.time()
    for i, p in enumerate(utility_prompts):
        if i % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [utility {i:4d}/{len(utility_prompts)}] {elapsed:6.1f}s")
        per_lf, n_tokens = compute_attribution_for_prompt(
            tok, model, saes, mean_lookup, layer_cand_lookup, p["prompt"], p["expected_continuation"])
        if n_tokens == 0:
            continue
        for (L, f), d in per_lf.items():
            accum_utility.append((L, f, p["prompt_id"], d, n_tokens))
    print(f"  utility done: {len(accum_utility)} firings in {time.time()-t0:.1f}s")

    # === Stage 2: validation (run before aggregation so fallback flag can be set) ===
    print("\n[stage 2] 200-pair real intervention validation")
    validation = run_validation(
        tok, model, saes, mean_lookup, accum_probe, accum_utility,
        probe_prompts, utility_prompts, n_pairs=N_VALIDATION_PAIRS)
    OUT_VALIDATION.parent.mkdir(parents=True, exist_ok=True)
    OUT_VALIDATION.write_text(json.dumps(validation, indent=2))
    print(f"  saved: {OUT_VALIDATION.name}")
    print(f"  Pearson r={validation['pearson_r']:.4f}  Spearman ρ={validation['spearman_rho']:.4f}")
    print(f"  rel_err_mean_filtered={validation['rel_err_mean_filtered']:.4f} "
          f"(p50={validation['rel_err_p50_filtered']:.4f}, p95={validation['rel_err_p95_filtered']:.4f}) "
          f"n_nontrivial={validation['n_nontrivial_pairs']}/{validation['n_validation_pairs']}")
    print(f"  rel_mae_global={validation['rel_mae_global']:.4f}  accept={validation['accept']}")

    # === Stage 3: aggregation ===
    print("\n[stage 3] aggregation per (layer, feature)")
    df_aggregate = aggregate_per_feature(
        accum_probe, accum_utility, cand_df, probe_prompts, validation["accept"])
    df_aggregate.to_parquet(OUT_PARQUET, index=False)
    print(f"  saved: {OUT_PARQUET.name} ({len(df_aggregate)} rows)")

    # === Stage 4: hash_log + sanity ===
    print("\n[stage 4] hash_log + sanity report")
    update_hash_log(df_aggregate, validation)
    print_sanity(df_aggregate)

    write_torch_peak(os.environ.get("SCHEDULER_JOB_ID"))
    print(f"\n[final VRAM] {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GiB")
    print("[done]")


if __name__ == "__main__":
    main()
