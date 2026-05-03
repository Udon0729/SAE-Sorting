"""Phase 3 v001.1 — ρ sweep sensitivity analysis (Pythia-160m).

Pre-registered design: docs/preregistrations/phase3_v001_1_rho_sweep.md (v2)

For each ρ ∈ RHO_SWEEP = [0.25, 0.5, 0.75]:
  - Linear attribution via single forward+backward per prompt (grad_dot computed
    once; per-ρ delta = (ρ - 1) · val · grad_dot is just a constant rescaling).
  - Aggregate to per (layer, feature) parquet:
    `phase3_causal_light_pythia_160m_rho{rho_tag}.parquet`

Real intervention validation:
  - Sample N_VALIDATION_PAIRS=500 (feature, prompt) pairs ONCE (seed=42)
  - For each ρ: run real intervention forward on the SAME pairs
  - Output per-ρ pearson/spearman + per-pair real TE arrays for cross-ρ analysis

C1-C5 採否判定は別スクリプト (`phase3_rho_sweep_decide.py`) で行う。本スクリプトは
データ生成のみで判定・採否は埋め込まない (decide 切離しで cherry picking 回避)。

Outputs:
  data/profiling/v001/phase3_causal_light_pythia_160m_rho025.parquet
  data/profiling/v001/phase3_causal_light_pythia_160m_rho050.parquet
  data/profiling/v001/phase3_causal_light_pythia_160m_rho075.parquet
  data/profiling/v001/phase3_rho_sweep_validation.json
  configs/hash_log.json updated (3 parquet entries + 1 validation entry)

Run via scheduler:
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
OUT_DIR = REPO_ROOT / "data" / "profiling" / "v001"
OUT_VALIDATION = OUT_DIR / "phase3_rho_sweep_validation.json"
HASH_LOG = REPO_ROOT / "configs" / "hash_log.json"

MODEL_REPO = "EleutherAI/pythia-160m-deduped"
MODEL_REVISION = "step143000"
SAE_REPO = "EleutherAI/sae-pythia-160m-deduped-32k"
DEVICE = "cuda:0"
DTYPE = torch.float32
MAX_TOKENS = 128

RHO_SWEEP = [0.25, 0.5, 0.75]
RHO_PHASE4_INPUT = 0.5  # pre-committed in pre-registration §3.4
N_VALIDATION_PAIRS = 500
VALIDATION_SEED = 42
REL_ERR_FLOOR = 1e-3

CATS = ["person_attribute", "geography", "organization", "occupation"]
PROBE_TYPES = {"positive", "paraphrase"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rho_tag(rho: float) -> str:
    return f"rho{int(round(rho * 100)):03d}"


def out_parquet_for_rho(rho: float) -> Path:
    return OUT_DIR / f"phase3_causal_light_pythia_160m_{rho_tag(rho)}.parquet"


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


def build_layer_cand_lookup(cand_df: pd.DataFrame, n_layers: int, num_latents: int) -> list[torch.Tensor]:
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
    layer_cand_lookup: list[torch.Tensor],
    prompt_text: str,
    answer_text: str,
    rho_sweep: list[float],
) -> tuple[dict[float, dict[tuple[int, int], float]], int]:
    """Returns (per_rho_per_lf, n_answer_tokens).
    per_rho_per_lf[ρ] maps (layer, feature_idx) -> sum over firing positions of ΔlogP linear
    under scale_down ρ intervention.
    """
    full_text = prompt_text.rstrip() + " " + answer_text.lstrip()
    enc_full = tok(full_text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS).to(DEVICE)
    enc_prompt = tok(prompt_text.rstrip(), return_tensors="pt", truncation=True, max_length=MAX_TOKENS).to(DEVICE)

    full_ids = enc_full.input_ids[0]
    n_total = full_ids.shape[0]
    n_prompt = enc_prompt.input_ids.shape[1]
    if n_prompt >= n_total:
        return {rho: {} for rho in rho_sweep}, 0
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

    per_rho_per_lf: dict[float, dict[tuple[int, int], float]] = {rho: {} for rho in rho_sweep}

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
            grad_dot = (active_W * active_g).sum(-1)  # ρ-independent

            base_units = (active_vals * grad_dot).tolist()
            active_idxs_list = active_idxs.tolist()

            for rho in rho_sweep:
                factor = rho - 1.0
                bucket = per_rho_per_lf[rho]
                for f, base in zip(active_idxs_list, base_units):
                    key = (L, f)
                    bucket[key] = bucket.get(key, 0.0) + factor * base

    return per_rho_per_lf, n_answer


# -- real intervention --------------------------------------------------------

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


# -- validation (per-ρ on shared pairs) --------------------------------------

def _per_rho_stats(linear_arr: np.ndarray, real_arr: np.ndarray) -> dict:
    if len(real_arr) > 2:
        pearson_r = float(pearsonr(real_arr, linear_arr).statistic)
        spearman_rho = float(spearmanr(real_arr, linear_arr).statistic)
    else:
        pearson_r = float("nan")
        spearman_rho = float("nan")
    nonzero_mask = np.abs(real_arr) > REL_ERR_FLOOR
    n_nonzero = int(nonzero_mask.sum())
    if n_nonzero > 0:
        rel_errs = np.abs(real_arr[nonzero_mask] - linear_arr[nonzero_mask]) / np.abs(real_arr[nonzero_mask])
        rel_err_mean = float(rel_errs.mean())
        rel_err_p50 = float(np.percentile(rel_errs, 50))
        rel_err_p95 = float(np.percentile(rel_errs, 95))
    else:
        rel_err_mean = float("nan")
        rel_err_p50 = float("nan")
        rel_err_p95 = float("nan")
    return {
        "n": int(len(real_arr)),
        "n_nontrivial": n_nonzero,
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "rel_err_mean_filtered": rel_err_mean,
        "rel_err_p50_filtered": rel_err_p50,
        "rel_err_p95_filtered": rel_err_p95,
    }


def run_validation(
    tok,
    model,
    saes: list[dict],
    accum_probe_per_rho: dict[float, list[tuple]],
    accum_utility_per_rho: dict[float, list[tuple]],
    probe_prompts: list[dict],
    utility_prompts: list[dict],
    n_pairs: int,
    rho_sweep: list[float],
) -> dict:
    """Sample n_pairs ONCE; run real intervention per ρ on shared pairs.
    Outputs per-ρ stats + same-pair real TE arrays for cross-ρ analysis.
    """
    rng = np.random.default_rng(VALIDATION_SEED)
    qid_to_probe = {p["qid"]: p for p in probe_prompts}
    pid_to_utility = {p["prompt_id"]: p for p in utility_prompts}

    # Build candidate sample pool: all (L, f, source, item_id) keys present in ρ=phase4_input
    # accumulator. Linear is proportional in ρ so the pool is identical across ρ; pick one.
    src_rho = RHO_PHASE4_INPUT if RHO_PHASE4_INPUT in rho_sweep else rho_sweep[0]
    samples: list[tuple] = []  # (L, f, prompt_text, answer_text, source, item_id)
    seen: set[tuple] = set()
    for L, f, qid, _ in accum_probe_per_rho[src_rho]:
        key = (int(L), int(f), "probe", qid)
        if key in seen:
            continue
        seen.add(key)
        p = qid_to_probe[qid]
        samples.append((int(L), int(f), p["prompt"], p["answer"], "probe", qid))
    for L, f, pid, _, _ in accum_utility_per_rho[src_rho]:
        key = (int(L), int(f), "utility", pid)
        if key in seen:
            continue
        seen.add(key)
        p = pid_to_utility[pid]
        samples.append((int(L), int(f), p["prompt"], p["expected_continuation"], "utility", pid))

    if len(samples) < n_pairs:
        n_pairs = len(samples)
    sample_indices = rng.choice(len(samples), size=n_pairs, replace=False).tolist()
    chosen = [samples[i] for i in sample_indices]

    # Build linear lookup per ρ
    accum_lookup: dict[float, dict[tuple, float]] = {rho: {} for rho in rho_sweep}
    for rho in rho_sweep:
        for L, f, qid, d in accum_probe_per_rho[rho]:
            accum_lookup[rho][(int(L), int(f), "probe", qid)] = d
        for L, f, pid, d, _ in accum_utility_per_rho[rho]:
            accum_lookup[rho][(int(L), int(f), "utility", pid)] = d

    per_rho_pairs: dict[float, dict[str, list]] = {
        rho: {"linear": [], "real": [], "skipped": 0} for rho in rho_sweep
    }
    pair_keys_used: list[dict] = []  # only pairs where ALL ρ succeeded
    pair_real_per_rho: dict[float, list[float]] = {rho: [] for rho in rho_sweep}

    t0 = time.time()
    for i, (L, f, prompt_text, answer_text, source, item_id) in enumerate(chosen):
        if i % 25 == 0:
            print(f"  [validation pair {i}/{n_pairs}] {time.time()-t0:.1f}s")
        per_rho_real: dict[float, float] = {}
        truncated = False
        for rho in rho_sweep:
            real_d = real_intervention_forward(tok, model, saes[L], L, f, rho, prompt_text, answer_text)
            if real_d is None:
                truncated = True
                per_rho_pairs[rho]["skipped"] += 1
                break
            per_rho_real[rho] = real_d
        if truncated:
            continue
        # All ρ succeeded; record
        pair_keys_used.append({"layer": int(L), "feature_idx": int(f), "source": source, "item_id": item_id})
        for rho in rho_sweep:
            real_d = per_rho_real[rho]
            linear_d = accum_lookup[rho].get((int(L), int(f), source, item_id), float("nan"))
            per_rho_pairs[rho]["linear"].append(linear_d)
            per_rho_pairs[rho]["real"].append(real_d)
            pair_real_per_rho[rho].append(real_d)

    per_rho_stats: dict = {}
    for rho in rho_sweep:
        linear_arr = np.array(per_rho_pairs[rho]["linear"], dtype=float)
        real_arr = np.array(per_rho_pairs[rho]["real"], dtype=float)
        s = _per_rho_stats(linear_arr, real_arr)
        s["n_skipped"] = int(per_rho_pairs[rho]["skipped"])
        per_rho_stats[str(rho)] = s

    return {
        "intervention_type": "scale_down_rho_sweep",
        "rho_sweep": list(rho_sweep),
        "rho_phase4_input_committed": RHO_PHASE4_INPUT,
        "n_validation_pairs_target": int(n_pairs),
        "n_validation_pairs_used_all_rho_succeed": int(len(pair_keys_used)),
        "rel_err_floor": REL_ERR_FLOOR,
        "validation_seed": VALIDATION_SEED,
        "per_rho": per_rho_stats,
        "shared_pairs": pair_keys_used,
        "real_te_per_pair_per_rho": {str(rho): pair_real_per_rho[rho] for rho in rho_sweep},
        "linear_te_per_pair_per_rho": {
            str(rho): per_rho_pairs[rho]["linear"] for rho in rho_sweep
        },
        "frozen_at": now_iso(),
    }


# -- aggregation --------------------------------------------------------------

def aggregate_per_feature(
    accum_probe: list[tuple],
    accum_utility: list[tuple],
    cand_df: pd.DataFrame,
    probe_prompts: list[dict],
    intervention_rho: float,
) -> pd.DataFrame:
    qid_to_cat = {p["qid"]: p["category"] for p in probe_prompts}

    df_probe = pd.DataFrame(accum_probe, columns=["layer", "feature_idx", "qid", "attribution"])
    df_probe["category"] = df_probe["qid"].map(qid_to_cat)

    df_util = pd.DataFrame(accum_utility, columns=["layer", "feature_idx", "prompt_id", "attribution", "n_target_tokens"])
    df_util["ud_per_token"] = -df_util["attribution"] / df_util["n_target_tokens"].clip(lower=1)

    g_probe = df_probe.groupby(["layer", "feature_idx", "category"])["attribution"].agg(["mean", "std", "count"]).reset_index()
    g_util = df_util.groupby(["layer", "feature_idx"])["ud_per_token"].agg(["mean", "std", "count"]).reset_index()
    g_util.columns = ["layer", "feature_idx", "ud_mean", "ud_std", "ud_count"]
    probe_idx = g_probe.set_index(["layer", "feature_idx", "category"])
    util_idx = g_util.set_index(["layer", "feature_idx"])

    label = f"scale_down_rho{intervention_rho}"

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
            "intervention_type": label,
            "intervention_rho": intervention_rho,
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
            "fallback_reason": None,  # set by phase3_rho_sweep_decide.py
        })

    return pd.DataFrame(rows)


# -- output -------------------------------------------------------------------

def update_hash_log(rho_sweep: list[float], validation: dict) -> None:
    log = json.loads(HASH_LOG.read_text()) if HASH_LOG.exists() else {}
    for rho in rho_sweep:
        out = out_parquet_for_rho(rho)
        log[f"phase3_causal_light_pythia_{rho_tag(rho)}_hash"] = {
            "file": str(out.relative_to(REPO_ROOT)),
            "sha256": sha256_of(out),
            "model": f"{MODEL_REPO}@{MODEL_REVISION}",
            "sae": SAE_REPO,
            "intervention_type": f"scale_down_rho{rho}",
            "intervention_rho": rho,
            "effect_estimator": "attribution_linear",
            "frozen_at": now_iso(),
        }
    log["phase3_rho_sweep_validation_hash"] = {
        "file": str(OUT_VALIDATION.relative_to(REPO_ROOT)),
        "sha256": sha256_of(OUT_VALIDATION),
        "rho_sweep": list(rho_sweep),
        "n_validation_pairs_used": validation["n_validation_pairs_used_all_rho_succeed"],
        "per_rho_pearson": {rho: validation["per_rho"][rho]["pearson_r"] for rho in validation["per_rho"]},
        "per_rho_spearman": {rho: validation["per_rho"][rho]["spearman_rho"] for rho in validation["per_rho"]},
        "frozen_at": now_iso(),
    }
    HASH_LOG.write_text(json.dumps(log, indent=2, ensure_ascii=False, sort_keys=True))


def print_per_rho_summary(rho_sweep: list[float], validation: dict) -> None:
    print()
    print("  Per-ρ validation summary:")
    print(f"  {'ρ':>5}  {'n':>5}  {'n_nz':>5}  {'pearson':>8}  {'spearman':>9}  {'rel_err_mean':>12}  {'rel_err_p50':>11}")
    for rho in rho_sweep:
        s = validation["per_rho"][str(rho)]
        print(f"  {rho:>5.2f}  {s['n']:>5d}  {s['n_nontrivial']:>5d}  "
              f"{s['pearson_r']:>8.4f}  {s['spearman_rho']:>9.4f}  "
              f"{s['rel_err_mean_filtered']:>12.4f}  {s['rel_err_p50_filtered']:>11.4f}")


def write_torch_peak(job_id: str | None) -> None:
    if not job_id:
        return
    target = REPO_ROOT / "ops" / "queue" / "logs" / f"{job_id}_torch_peak.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"0": int(torch.cuda.max_memory_allocated(0))}))


# -- main ---------------------------------------------------------------------

def main() -> None:
    print(f"--- Phase 3 v001.1 ρ sweep (Pythia-160m) ---")
    print(f"  device:    {torch.cuda.get_device_name(0)}")
    print(f"  torch:     {torch.__version__}")
    print(f"  RHO_SWEEP: {RHO_SWEEP}")
    print(f"  N_VALIDATION_PAIRS: {N_VALIDATION_PAIRS}")

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
    print(f"  probe: {len(probe_prompts)}")

    print("[load] D_utility")
    utility_prompts = load_utility_prompts()
    print(f"  utility: {len(utility_prompts)}")

    # === Stage 1: attribution loop (per ρ accumulators) ===
    print("\n[stage 1] attribution loop (probe)")
    accum_probe_per_rho: dict[float, list[tuple]] = {rho: [] for rho in RHO_SWEEP}
    t0 = time.time()
    for i, p in enumerate(probe_prompts):
        if i % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            print(f"  [probe {i:4d}/{len(probe_prompts)}] {elapsed:6.1f}s ({rate:.1f}/s)")
        per_rho_per_lf, _ = compute_attribution_for_prompt(
            tok, model, saes, layer_cand_lookup, p["prompt"], p["answer"], RHO_SWEEP)
        for rho in RHO_SWEEP:
            for (L, f), d in per_rho_per_lf[rho].items():
                accum_probe_per_rho[rho].append((L, f, p["qid"], d))
    print(f"  probe done in {time.time()-t0:.1f}s")
    for rho in RHO_SWEEP:
        print(f"    ρ={rho}: {len(accum_probe_per_rho[rho])} firings")

    print("\n[stage 1b] attribution loop (utility)")
    accum_utility_per_rho: dict[float, list[tuple]] = {rho: [] for rho in RHO_SWEEP}
    t0 = time.time()
    for i, p in enumerate(utility_prompts):
        if i % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [utility {i:4d}/{len(utility_prompts)}] {elapsed:6.1f}s")
        per_rho_per_lf, n_tokens = compute_attribution_for_prompt(
            tok, model, saes, layer_cand_lookup, p["prompt"], p["expected_continuation"], RHO_SWEEP)
        if n_tokens == 0:
            continue
        for rho in RHO_SWEEP:
            for (L, f), d in per_rho_per_lf[rho].items():
                accum_utility_per_rho[rho].append((L, f, p["prompt_id"], d, n_tokens))
    print(f"  utility done in {time.time()-t0:.1f}s")

    # === Stage 2: validation ===
    print(f"\n[stage 2] {N_VALIDATION_PAIRS}-pair real intervention validation × {len(RHO_SWEEP)} ρ (shared pairs)")
    validation = run_validation(
        tok, model, saes,
        accum_probe_per_rho, accum_utility_per_rho,
        probe_prompts, utility_prompts,
        n_pairs=N_VALIDATION_PAIRS, rho_sweep=RHO_SWEEP)
    OUT_VALIDATION.parent.mkdir(parents=True, exist_ok=True)
    OUT_VALIDATION.write_text(json.dumps(validation, indent=2))
    print(f"  saved: {OUT_VALIDATION.name}")
    print_per_rho_summary(RHO_SWEEP, validation)

    # === Stage 3: aggregation per ρ ===
    print("\n[stage 3] aggregation per (layer, feature) × per ρ")
    for rho in RHO_SWEEP:
        df = aggregate_per_feature(
            accum_probe_per_rho[rho], accum_utility_per_rho[rho], cand_df, probe_prompts, rho)
        out = out_parquet_for_rho(rho)
        df.to_parquet(out, index=False)
        print(f"  ρ={rho}: saved {out.name} ({len(df)} rows)")

    # === Stage 4: hash_log ===
    print("\n[stage 4] hash_log update")
    update_hash_log(RHO_SWEEP, validation)
    print(f"  hash_log updated with 3 parquet entries + 1 validation entry")

    write_torch_peak(os.environ.get("SCHEDULER_JOB_ID"))
    print(f"\n[final VRAM] {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GiB")
    print("[done] phase3_rho_sweep_decide.py で C1-C5 判定を行うこと")


if __name__ == "__main__":
    main()
