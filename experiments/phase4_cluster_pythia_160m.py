"""Phase 4 v001 — z_f build + HDBSCAN clustering + cluster-level heavy profiling.

Pre-registered design: docs/preregistrations/phase4_v001.md

Stages (all in one GPU job; clustering is CPU but co-located for simplicity):

  Stage A: build z_f (14 dim) from phase3 v2 + phase1
  Stage B: HDBSCAN cluster
  Stage C: top_activating_examples per cluster (top 50 D_probe prompts)
  Stage D: heavy profiling per cluster:
              7 interventions H ∈ {zero, mean, scale_down ρ∈{.25,.5,.75}, neg_scale ρ∈{.25,.5}}
              50 D_probe (cluster's primary_category, seed=42 sample) + 48 D_utility (8 per subset)
              metrics: ΔlogP_target / Δlog_perplexity / generation_collapse_rate

Outputs:
  data/profiling/v001/phase4_feature_vectors.parquet
  data/profiling/v001/phase4_cluster_assignment.parquet
  data/profiling/v001/phase4_top_activating_examples.jsonl
  data/profiling/v001/phase4_cluster_heavy_profile.parquet
  configs/hash_log.json updated (4 entries)

Decision (G1-G5) is in phase4_decide.py. Labeling is in phase4_label_pythia_160m.py.

Run via scheduler:
  command:        uv run python experiments/phase4_cluster_pythia_160m.py
  requested_gpus: [0]
  vram_budget_gib: 16
  env: { SCHEDULER_JOB_ID: "<job_id>" }
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from phase3_attribution_pythia_160m import (  # noqa: E402
    CATS,
    MODEL_REPO,
    MODEL_REVISION,
    SAE_REPO,
    DEVICE,
    DTYPE,
    MAX_TOKENS,
    load_all_saes,
    load_pythia,
    load_probe_prompts,
    load_utility_prompts,
    write_torch_peak,
)

DATA_DIR = REPO_ROOT / "data" / "profiling" / "v001"
PHASE3_V2 = DATA_DIR / "phase3_causal_light_pythia_160m_rho050_v2.parquet"
PHASE1 = DATA_DIR / "phase1_features.parquet"
OUT_FV = DATA_DIR / "phase4_feature_vectors.parquet"
OUT_CLUSTER = DATA_DIR / "phase4_cluster_assignment.parquet"
OUT_TOP_EX = DATA_DIR / "phase4_top_activating_examples.jsonl"
OUT_HEAVY = DATA_DIR / "phase4_cluster_heavy_profile.parquet"
HASH_LOG = REPO_ROOT / "configs" / "hash_log.json"
PROBE_PROMPTS_PATH = REPO_ROOT / "data" / "probe" / "v001" / "prompts.jsonl"

# ============== pre-committed config (phase4_v001.md) ==============
LAMBDA_S = 1.0
LAMBDA_C = 2.0
LAMBDA_U = 2.0
UTILITY_SUBSETS = ["U1", "U2", "U3", "U4", "U5", "U6"]

HDBSCAN_MIN_CLUSTER_SIZE = 50
HDBSCAN_MIN_SAMPLES = 10
HDBSCAN_METRIC = "euclidean"
HDBSCAN_CLUSTER_SELECTION = "eom"
RANDOM_SEED = 42

# Heavy interventions (pre-registered §3.3)
HEAVY_INTERVENTIONS: list[dict] = [
    {"id": "H1", "type": "zero",         "param": None},
    {"id": "H2", "type": "mean",         "param": None},
    {"id": "H3", "type": "scale_down",   "param": 0.25},
    {"id": "H4", "type": "scale_down",   "param": 0.50},
    {"id": "H5", "type": "scale_down",   "param": 0.75},
    {"id": "H6", "type": "negative_scale","param": 0.25},
    {"id": "H7", "type": "negative_scale","param": 0.50},
]

N_PROBE_PER_CLUSTER = 50
N_UTILITY_PER_SUBSET = 8     # 8 × 6 subsets = 48
GEN_MAX_NEW_TOKENS = 32

# generation_collapse_rate (§3.5)
COLLAPSE_LENGTH_THRESH = 0.5
COLLAPSE_REPETITION_NGRAM = 4
COLLAPSE_REPETITION_MAX_COUNT = 5
COLLAPSE_EMPTY_TOKEN_THRESH = 5
COLLAPSE_FORMAT_PUNCT_RATIO = 0.5
COLLAPSE_PUNCT_CHARS = set('.,:;?!\n')


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============== Stage A: build z_f ==================================

def build_feature_vectors(phase3_v2: pd.DataFrame, phase1: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Returns (df with key + raw + z columns, list of z column names in order)."""
    p1_cols = ["layer", "feature_idx", "overall_firing_rate"] + [f"firing_rate_{c}" for c in CATS]
    p1 = phase1[p1_cols].copy()

    p3_cols = (["layer", "feature_idx"]
               + [f"delta_m_target_per_cat_{c}" for c in CATS]
               + [f"delta_m_utility_subset_{u}" for u in UTILITY_SUBSETS])
    p3 = phase3_v2[p3_cols].copy()

    df = p3.merge(p1, on=["layer", "feature_idx"], how="left")
    if df[p1_cols[2:]].isna().any().any():
        n_bad = int(df[p1_cols[2:]].isna().any(axis=1).sum())
        raise RuntimeError(f"{n_bad} candidate (layer, feature) rows missing phase1 firing stats")

    # s_f[i] = log1p(firing_rate_cat_i / max(overall_firing_rate, 1e-9))
    overall = df["overall_firing_rate"].clip(lower=1e-9)
    s_cols = []
    for c in CATS:
        col = f"s_{c}"
        df[col] = np.log1p(df[f"firing_rate_{c}"] / overall)
        s_cols.append(col)

    c_cols = []
    for c in CATS:
        col = f"c_{c}"
        df[col] = df[f"delta_m_target_per_cat_{c}"]
        c_cols.append(col)

    u_cols = []
    for u in UTILITY_SUBSETS:
        col = f"u_{u}"
        df[col] = df[f"delta_m_utility_subset_{u}"]
        u_cols.append(col)

    raw_cols = s_cols + c_cols + u_cols  # 14
    if df[raw_cols].isna().any().any():
        n_bad = int(df[raw_cols].isna().any(axis=1).sum())
        print(f"  [warn] {n_bad} rows have NaN in z_f raw columns; filling with 0 (no firing observed)")
        df[raw_cols] = df[raw_cols].fillna(0.0)

    # per-axis z-score, then apply λ
    weight_per_col = ([LAMBDA_S] * 4) + ([LAMBDA_C] * 4) + ([LAMBDA_U] * 6)
    z_cols = []
    for col, w in zip(raw_cols, weight_per_col):
        v = df[col].astype(np.float64).to_numpy()
        mu = float(v.mean())
        sigma = float(v.std(ddof=0))
        if sigma < 1e-9:
            sigma = 1e-9
        z = w * (v - mu) / sigma
        z_col = f"z_{col}"
        df[z_col] = z.astype(np.float32)
        z_cols.append(z_col)

    keep = ["layer", "feature_idx"] + raw_cols + z_cols
    return df[keep].reset_index(drop=True), z_cols


# ============== Stage B: HDBSCAN ====================================

def hdbscan_cluster(fv_df: pd.DataFrame, z_cols: list[str]) -> pd.DataFrame:
    np.random.seed(RANDOM_SEED)
    X = fv_df[z_cols].to_numpy(dtype=np.float64)
    print(f"  HDBSCAN over {X.shape[0]} × {X.shape[1]}")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
        cluster_selection_method=HDBSCAN_CLUSTER_SELECTION,
        prediction_data=True,
        allow_single_cluster=False,
        core_dist_n_jobs=1,
    )
    labels = clusterer.fit_predict(X)
    probs = clusterer.probabilities_
    out = fv_df[["layer", "feature_idx"]].copy()
    out["cluster_id"] = labels.astype(np.int32)
    out["cluster_membership_prob"] = probs.astype(np.float32)
    return out


# ============== Stage C: top_activating_examples =====================

def build_top_activating_examples(
    cluster_assn: pd.DataFrame,
    phase1: pd.DataFrame,
    probe_prompts: list[dict],
) -> list[dict]:
    """Per cluster, gather member features' top_qids/top_values, sum activations
    by qid, return top 50 unique prompts."""
    qid_to_prompt = {p["qid"]: p for p in probe_prompts}
    p1_idx = phase1.set_index(["layer", "feature_idx"])[["top_qids", "top_values"]]
    out: list[dict] = []
    for cid in sorted(cluster_assn["cluster_id"].unique()):
        if cid < 0:
            continue
        members = cluster_assn[cluster_assn["cluster_id"] == cid][["layer", "feature_idx"]]
        agg: Counter = Counter()
        for _, m in members.iterrows():
            try:
                row = p1_idx.loc[(int(m["layer"]), int(m["feature_idx"]))]
            except KeyError:
                continue
            qids = list(row["top_qids"])
            vals = list(row["top_values"])
            for q, v in zip(qids, vals):
                agg[q] += float(v)
        top = agg.most_common(50)
        examples = []
        for qid, score in top:
            p = qid_to_prompt.get(qid)
            examples.append({
                "qid": qid,
                "cum_activation": score,
                "category": (p or {}).get("category"),
                "prompt_type": (p or {}).get("prompt_type"),
                "prompt": (p or {}).get("prompt"),
                "answer": (p or {}).get("answer"),
            })
        out.append({
            "cluster_id": int(cid),
            "n_members": int(len(members)),
            "examples": examples,
        })
    return out


# ============== Stage D: heavy profiling ============================

def derive_primary_category(cluster_assn: pd.DataFrame, fv_df: pd.DataFrame) -> dict[int, str]:
    """primary_category(c) = argmax_i mean(c_f[i] over members of c)."""
    merged = cluster_assn.merge(fv_df, on=["layer", "feature_idx"])
    out: dict[int, str] = {}
    for cid, sub in merged.groupby("cluster_id"):
        if cid < 0:
            continue
        means = {c: float(sub[f"c_{c}"].mean()) for c in CATS}
        out[int(cid)] = max(means, key=means.get)
    return out


def sample_prompts_for_cluster(
    primary_cat: str,
    probe_prompts: list[dict],
    utility_prompts: list[dict],
    cid: int,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(RANDOM_SEED + cid)
    # D_probe: positive+paraphrase of primary_cat
    eligible_probe = [p for p in probe_prompts if p["category"] == primary_cat]
    rng.shuffle(eligible_probe)
    probe_sample = eligible_probe[:N_PROBE_PER_CLUSTER]
    # D_utility: stratified 8 per subset
    util_by_subset: dict[str, list] = {u: [] for u in UTILITY_SUBSETS}
    for p in utility_prompts:
        if p["category_id"] in util_by_subset:
            util_by_subset[p["category_id"]].append(p)
    util_sample: list[dict] = []
    for u in UTILITY_SUBSETS:
        items = util_by_subset[u][:]
        rng.shuffle(items)
        util_sample.extend(items[:N_UTILITY_PER_SUBSET])
    return probe_sample, util_sample


def make_cluster_intervention_hook(
    sae: dict,
    features: list[int],          # cluster member feature_idx values at this layer
    feature_means: dict[int, float],  # mean_value per feature (Phase 1)
    intervention: dict,
):
    """Returns a forward_hook that applies `intervention` to all `features` simultaneously
    at firing positions (per-prompt SAE encode → top-k → match → write-back).

    Vectorized: builds a per-latent mask (num_latents,) and a per-latent mean lookup, then
    a single matmul on firing positions writes the residual delta — no Python loop over
    cluster members at runtime."""
    weights = sae["weights"]
    cfg = sae["cfg"]
    K = int(cfg["k"])
    signed = bool(cfg.get("signed", False))
    num_latents = int(weights["encoder.weight"].shape[0])
    int_type = intervention["type"]
    rho = intervention["param"]

    feat_mask = torch.zeros(num_latents, dtype=torch.bool, device=DEVICE)
    feat_mask[torch.tensor(features, dtype=torch.long, device=DEVICE)] = True
    mean_lookup = torch.zeros(num_latents, dtype=DTYPE, device=DEVICE) if int_type == "mean" else None
    if mean_lookup is not None:
        for f, mv in feature_means.items():
            mean_lookup[f] = float(mv)

    def hook(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        sae_in = h[0] - weights["b_dec"]
        pre = sae_in @ weights["encoder.weight"].T + weights["encoder.bias"]
        topk = pre.topk(K, dim=-1)
        vals = topk.values
        if not signed:
            vals = vals.relu()
        idxs = topk.indices  # (T, K)

        member = feat_mask[idxs]               # (T, K) bool
        if not member.any():
            return output

        # Coordinates of firing (t, k) where (t, k) is a cluster member
        coords = member.nonzero(as_tuple=False)  # (N, 2)
        t_idx = coords[:, 0]
        k_idx = coords[:, 1]
        cur_vals = vals[t_idx, k_idx]            # (N,)
        feat_at = idxs[t_idx, k_idx]             # (N,) actual feature indices

        if int_type == "zero":
            new_vals = torch.zeros_like(cur_vals)
        elif int_type == "mean":
            new_vals = mean_lookup[feat_at]
        elif int_type == "scale_down":
            new_vals = float(rho) * cur_vals
        elif int_type == "negative_scale":
            new_vals = -float(rho) * cur_vals
        else:
            raise ValueError(int_type)

        delta_scalar = (new_vals - cur_vals)     # (N,)
        W_dec_active = weights["W_dec"][feat_at] # (N, d)
        d_per_pos = delta_scalar.unsqueeze(-1) * W_dec_active  # (N, d)

        h_new = h.clone()
        # scatter-add: multiple (t, k) entries at the same t are summed
        h_new[0].index_add_(0, t_idx, d_per_pos)
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new

    return hook


def detect_collapse(base_text: str, int_text: str, base_token_count: int, int_token_count: int) -> bool:
    if int_token_count < COLLAPSE_EMPTY_TOKEN_THRESH:
        return True
    if base_token_count >= 1:
        rel = abs(int_token_count - base_token_count) / max(base_token_count, 1)
        if rel > COLLAPSE_LENGTH_THRESH:
            return True
    # repetition: 4-gram count over int generated tokens (string-split fallback)
    words = int_text.split()
    if len(words) >= COLLAPSE_REPETITION_NGRAM:
        ngrams = [tuple(words[i:i + COLLAPSE_REPETITION_NGRAM])
                  for i in range(len(words) - COLLAPSE_REPETITION_NGRAM + 1)]
        c = Counter(ngrams)
        if c and c.most_common(1)[0][1] > COLLAPSE_REPETITION_MAX_COUNT:
            return True
    base_punct = sum(1 for ch in base_text if ch in COLLAPSE_PUNCT_CHARS)
    int_punct = sum(1 for ch in int_text if ch in COLLAPSE_PUNCT_CHARS)
    if base_punct >= 2 and int_punct < COLLAPSE_FORMAT_PUNCT_RATIO * base_punct:
        return True
    return False


def measure_logp(model, tok, prompt: str, target: str) -> tuple[float, int]:
    """logP(target | prompt) summed, plus n_target_tokens."""
    full = prompt.rstrip() + " " + target.lstrip()
    enc_full = tok(full, return_tensors="pt", truncation=True, max_length=MAX_TOKENS).to(DEVICE)
    enc_p = tok(prompt.rstrip(), return_tensors="pt", truncation=True, max_length=MAX_TOKENS).to(DEVICE)
    n_total = enc_full.input_ids.shape[1]
    n_p = enc_p.input_ids.shape[1]
    if n_p >= n_total:
        return 0.0, 0
    answer_ids = enc_full.input_ids[0, n_p:n_total]
    pred_pos = torch.arange(n_p - 1, n_total - 1, device=DEVICE)
    with torch.no_grad():
        out = model(enc_full.input_ids, use_cache=False)
    lp = F.log_softmax(out.logits[0, pred_pos], dim=-1).gather(1, answer_ids.unsqueeze(-1)).sum().item()
    return float(lp), int(n_total - n_p)


def measure_generation(model, tok, prompt: str) -> tuple[str, int]:
    enc = tok(prompt.rstrip(), return_tensors="pt", truncation=True, max_length=MAX_TOKENS).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            enc.input_ids,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tok.eos_token_id,
            use_cache=True,
        )
    new_tokens = out[0, enc.input_ids.shape[1]:]
    text = tok.decode(new_tokens, skip_special_tokens=True)
    return text, int(new_tokens.shape[0])


def heavy_profile_clusters(
    tok,
    model,
    saes: list[dict],
    cluster_assn: pd.DataFrame,
    fv_df: pd.DataFrame,
    feature_means_per_layer: dict[tuple[int, int], float],
    probe_prompts: list[dict],
    utility_prompts: list[dict],
) -> pd.DataFrame:
    primary_cats = derive_primary_category(cluster_assn, fv_df)
    cluster_ids = sorted(c for c in primary_cats.keys() if c >= 0)
    print(f"  heavy profiling {len(cluster_ids)} clusters × {len(HEAVY_INTERVENTIONS)} H + 1 baseline")

    rows: list[dict] = []
    t_start = time.time()
    for cidx, cid in enumerate(cluster_ids):
        primary_cat = primary_cats[cid]
        members = cluster_assn[cluster_assn["cluster_id"] == cid][["layer", "feature_idx"]]
        # group features by layer
        by_layer: dict[int, list[int]] = {}
        for _, m in members.iterrows():
            by_layer.setdefault(int(m["layer"]), []).append(int(m["feature_idx"]))

        probe_sample, util_sample = sample_prompts_for_cluster(primary_cat, probe_prompts, utility_prompts, cid)

        # === baseline (no hook) ===
        base_logp_probe = []
        base_logp_util = []
        base_gen_text = []
        base_gen_tokens = []
        for p in probe_sample:
            lp, _ = measure_logp(model, tok, p["prompt"], p["answer"])
            base_logp_probe.append(lp)
        for p in util_sample:
            lp, n_t = measure_logp(model, tok, p["prompt"], p["expected_continuation"])
            base_logp_util.append((lp, n_t))
            text, n_g = measure_generation(model, tok, p["prompt"])
            base_gen_text.append(text)
            base_gen_tokens.append(n_g)

        # === per H intervention ===
        for H in HEAVY_INTERVENTIONS:
            handles = []
            for L, feats in by_layer.items():
                fm = {f: feature_means_per_layer.get((L, f), 0.0) for f in feats}
                hook = make_cluster_intervention_hook(saes[L], feats, fm, H)
                handles.append(model.gpt_neox.layers[L].register_forward_hook(hook))
            try:
                # D_probe: ΔlogP_target per cat decomposition
                int_logp_probe = []
                for p in probe_sample:
                    lp, _ = measure_logp(model, tok, p["prompt"], p["answer"])
                    int_logp_probe.append(lp)
                # D_utility: log-perplexity + generation
                int_logp_util = []
                int_gen_collapse = []
                for j, p in enumerate(util_sample):
                    lp, n_t = measure_logp(model, tok, p["prompt"], p["expected_continuation"])
                    int_logp_util.append((lp, n_t))
                    text, n_g = measure_generation(model, tok, p["prompt"])
                    coll = detect_collapse(base_gen_text[j], text, base_gen_tokens[j], n_g)
                    int_gen_collapse.append(int(coll))
            finally:
                for h in handles:
                    h.remove()

            # D_probe TE: sum across primary_cat prompts (positive direction = ΔlogP > 0 doesn't make sense
            # but we compute mean ΔlogP; large NEGATIVE means cluster ablation hurts target)
            te_probe_arr = np.array([i - b for i, b in zip(int_logp_probe, base_logp_probe)], dtype=float)
            te_dprobe = float(te_probe_arr.mean())

            # D_utility ud per token: -ΔlogP / n
            ud_arr = []
            for (b_lp, n_t_b), (i_lp, n_t_i) in zip(base_logp_util, int_logp_util):
                if n_t_b == 0:
                    continue
                ud_arr.append(-(i_lp - b_lp) / n_t_b)
            ud_arr = np.array(ud_arr, dtype=float) if ud_arr else np.array([0.0])
            ud_perplexity = float(ud_arr.mean())
            gen_collapse = float(np.mean(int_gen_collapse)) if int_gen_collapse else 0.0

            # per-subset UD breakdown
            ud_per_subset: dict[str, float] = {}
            for sub in UTILITY_SUBSETS:
                sub_arr = []
                for j, p in enumerate(util_sample):
                    if p["category_id"] != sub:
                        continue
                    b_lp, n_t_b = base_logp_util[j]
                    i_lp, n_t_i = int_logp_util[j]
                    if n_t_b == 0:
                        continue
                    sub_arr.append(-(i_lp - b_lp) / n_t_b)
                ud_per_subset[sub] = float(np.mean(sub_arr)) if sub_arr else float("nan")

            rows.append({
                "cluster_id": cid,
                "primary_category": primary_cat,
                "n_members": int(len(members)),
                "intervention_id": H["id"],
                "intervention_type": H["type"],
                "intervention_param": H["param"] if H["param"] is not None else float("nan"),
                "te_dprobe": te_dprobe,
                "ud_perplexity": ud_perplexity,
                "gen_collapse_rate": gen_collapse,
                "ud_subset_U1": ud_per_subset["U1"],
                "ud_subset_U2": ud_per_subset["U2"],
                "ud_subset_U3": ud_per_subset["U3"],
                "ud_subset_U4": ud_per_subset["U4"],
                "ud_subset_U5": ud_per_subset["U5"],
                "ud_subset_U6": ud_per_subset["U6"],
                "n_probe_prompts": int(len(probe_sample)),
                "n_utility_prompts": int(len(util_sample)),
            })

        elapsed = time.time() - t_start
        per_cluster = elapsed / max(cidx + 1, 1)
        eta = per_cluster * (len(cluster_ids) - cidx - 1)
        if cidx < 3 or cidx % 5 == 0 or cidx + 1 == len(cluster_ids):
            print(f"  [cluster {cidx + 1:3d}/{len(cluster_ids)}] cid={cid} primary={primary_cat} "
                  f"members={len(members)}  elapsed={elapsed:6.1f}s  eta={eta:6.1f}s")

    return pd.DataFrame(rows)


# ============== output / hash log ====================================

def update_hash_log() -> None:
    log = json.loads(HASH_LOG.read_text()) if HASH_LOG.exists() else {}
    for tag, path in [
        ("phase4_feature_vectors_hash", OUT_FV),
        ("phase4_cluster_assignment_hash", OUT_CLUSTER),
        ("phase4_top_activating_examples_hash", OUT_TOP_EX),
        ("phase4_cluster_heavy_profile_hash", OUT_HEAVY),
    ]:
        log[tag] = {
            "file": str(path.relative_to(REPO_ROOT)),
            "sha256": sha256_of(path),
            "model": f"{MODEL_REPO}@{MODEL_REVISION}",
            "sae": SAE_REPO,
            "frozen_at": now_iso(),
        }
    HASH_LOG.write_text(json.dumps(log, indent=2, ensure_ascii=False, sort_keys=True))


def main() -> None:
    print(f"--- Phase 4 v001 (Pythia-160m) — cluster + heavy profiling ---")
    print(f"  device: {torch.cuda.get_device_name(0)}")
    print(f"  HDBSCAN min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE} min_samples={HDBSCAN_MIN_SAMPLES}")
    print(f"  λ = ({LAMBDA_S}, {LAMBDA_C}, {LAMBDA_U})")
    print(f"  H interventions: {[H['id'] for H in HEAVY_INTERVENTIONS]}")

    print("[load] phase3 v2 + phase1")
    phase3_v2 = pd.read_parquet(PHASE3_V2)
    phase1 = pd.read_parquet(PHASE1)
    print(f"  phase3 v2: {len(phase3_v2)} rows, phase1: {len(phase1)} rows")

    print("[stage A] build z_f")
    fv_df, z_cols = build_feature_vectors(phase3_v2, phase1)
    fv_df.to_parquet(OUT_FV, index=False)
    print(f"  saved {OUT_FV.name} ({len(fv_df)} rows × {len(fv_df.columns)} cols)")

    print("[stage B] HDBSCAN clustering")
    cluster_assn = hdbscan_cluster(fv_df, z_cols)
    cluster_assn.to_parquet(OUT_CLUSTER, index=False)
    n_clusters = int((cluster_assn["cluster_id"] >= 0).sum() and cluster_assn[cluster_assn.cluster_id >= 0].cluster_id.nunique())
    n_noise = int((cluster_assn["cluster_id"] == -1).sum())
    print(f"  saved {OUT_CLUSTER.name}: {n_clusters} clusters, {n_noise}/{len(cluster_assn)} noise")
    if n_clusters == 0:
        print("  [G1 fail] no valid HDBSCAN clusters — abort heavy profiling, write empty heavy parquet")
        empty_heavy = pd.DataFrame(columns=[
            "cluster_id", "primary_category", "n_members", "intervention_id", "intervention_type",
            "intervention_param", "te_dprobe", "ud_perplexity", "gen_collapse_rate",
        ])
        empty_heavy.to_parquet(OUT_HEAVY, index=False)
        OUT_TOP_EX.write_text("")
        update_hash_log()
        write_torch_peak(os.environ.get("SCHEDULER_JOB_ID"))
        return

    print("[load] D_probe + D_utility")
    probe_prompts = load_probe_prompts()
    utility_prompts = load_utility_prompts()
    print(f"  probe={len(probe_prompts)} utility={len(utility_prompts)}")

    print("[stage C] top_activating_examples per cluster")
    top_ex = build_top_activating_examples(cluster_assn, phase1, probe_prompts)
    with OUT_TOP_EX.open("w", encoding="utf-8") as f:
        for entry in top_ex:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  saved {OUT_TOP_EX.name} ({len(top_ex)} clusters)")

    print("[load] Pythia + 12 SAEs (for heavy profiling)")
    tok, model = load_pythia()
    n_layers = model.config.num_hidden_layers
    saes = load_all_saes(n_layers)
    print(f"  param VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GiB")

    print("[load] feature mean values from phase1")
    p1_means = phase1.set_index(["layer", "feature_idx"])["mean_value"].to_dict()

    print("[stage D] heavy profiling")
    t0 = time.time()
    heavy_df = heavy_profile_clusters(
        tok, model, saes, cluster_assn, fv_df, p1_means, probe_prompts, utility_prompts)
    heavy_df.to_parquet(OUT_HEAVY, index=False)
    print(f"  heavy done in {time.time()-t0:.1f}s, saved {OUT_HEAVY.name} ({len(heavy_df)} rows)")

    print("[hash_log] update")
    update_hash_log()
    print("  4 entries updated")

    write_torch_peak(os.environ.get("SCHEDULER_JOB_ID"))
    print(f"\n[final VRAM] {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GiB")
    print("[done] phase4_label + phase4_decide で 5-class label と G1-G5 判定")


if __name__ == "__main__":
    main()
