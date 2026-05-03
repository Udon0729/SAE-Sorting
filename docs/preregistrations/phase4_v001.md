# Pre-registration: Phase 4 v001 — cluster + heavy profiling + 5-class labeling

**Pre-registered**: 2026-05-03 (UTC)
**Author**: K. Munaoka with Claude Code
**Predecessor**: Phase 3 v001.1 (`docs/preregistrations/phase3_v001_1_rho_sweep.md`) → ALLOW、ρ=0.5 parquet を Phase 4 入力に commit 済

## 1. 背景

Phase 3 v001.1 で attribution patching の linear approximation が ρ=0.5 setup で十分高精度 (Pearson 0.92, Spearman 0.93+) と確認された。本 Phase 4 は、得られた per-(layer, feature) 因果指標を入力に:

1. **z_f 14-dim 特徴ベクトル** 構築 (semantic selectivity + per-cat causal TE + per-utility-subset damage)
2. **HDBSCAN** で feature クラスター発見
3. **cluster 単位の heavy intervention** で full causal profile 構築
4. **5-class taxonomy label** 付与 (KNOWLEDGE-ASSOCIATED / ENTANGLED / INFRASTRUCTURE / UNSTABLE / UNKNOWN)

を実行する。これは本研究の **概念的ゴール (5-class taxonomy 自動発見)** を初めて実体化する milestone である。

post-hoc に HDBSCAN params を試行錯誤・5-class threshold を結果見て調整 = researcher degrees of freedom (cherry picking 同種)。**Phase 3 v001.1 同水準の規律** で実験前に全 design choice を pre-commit する。

## 2. v001 scope (確定事項)

| 項目 | v001 | v002 deferred |
|---|---|---|
| model | **Pythia-160m only** | Qwen3-8B Phase 4 |
| 入力 | `data/profiling/v001/phase3_causal_light_pythia_160m_rho050_v2.parquet` (Step 4.1 で再集約後の 10,311 candidates) | full feature run |
| categories | **4** (D_probe v001 native: person_attribute / geography / organization / occupation) | 8 cat 拡張 |
| utility subsets | **6** (D_utility v001 native: U1-U6) | – |
| **z_f dim** | **14** = 4 (s_f) + 4 (c_f) + 6 (u_f) | spec 30 dim (8 cat × 3 + 6 util) |
| n_f (NTE per cat) | **clustering から除外**: 4 cat では n_f[i]=(Σc_f − c_f[i])/3 で c_f に rank-1 redundant (parquet には documentation 用に保存) | 8 cat なら独立性回復、clustering に追加 |
| heavy interventions | spec 完全準拠: zero / mean / scale_down ρ∈{0.25,0.5,0.75} / negative_scale ρ∈{0.25,0.5} = **7 active** | – |

## 3. Pre-committed design choices

### 3.1 z_f 構成 (14 dim)

```
z_f = [λ_s · s_f (4) | λ_c · c_f (4) | λ_u · u_f (6)]
λ_s = 1.0, λ_c = 2.0, λ_u = 2.0    (implementation.md §8.4 spec)

s_f[i] = log1p(firing_rate_cat_i / max(overall_firing_rate, 1e-9))   # i ∈ {person_attribute, geography, organization, occupation}
c_f[i] = delta_m_target_per_cat_<cat_i>                              # phase3 ρ=0.5 parquet
u_f[k] = delta_m_utility_subset_U<k>                                 # k ∈ {1, 2, 3, 4, 5, 6} (Step 4.1 で生成)

per-axis z-score: μ_axis=0, σ_axis=1 over 全 10,311 行
σ < 1e-9 軸は σ=1e-9 で割る (degenerate axis のフェイルセーフ)
```

### 3.2 HDBSCAN パラメータ (implementation.md §8.5 準拠)

```python
HDBSCAN(
    min_cluster_size=50,
    min_samples=10,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True,
    allow_single_cluster=False,
)
random_state = 42  (numpy global)
no PCA / no UMAP                   # 14 dim そのまま
```

### 3.3 Heavy intervention schedule (implementation.md §8.6 準拠)

| ID | type | param | val 書換え規則 |
|---|---|---|---|
| H1 | zero ablation | – | val := 0 |
| H2 | mean replacement | – | val := E[val | f firing] (Phase 1 mean_value) |
| H3 | scale_down | ρ=0.25 | val := 0.25 · val |
| H4 | scale_down | ρ=0.5 | val := 0.5 · val |
| H5 | scale_down | ρ=0.75 | val := 0.75 · val |
| H6 | negative_scale | ρ=0.25 | val := -0.25 · val |
| H7 | negative_scale | ρ=0.5 | val := -0.5 · val |

(scale_down ρ=1.0 = no-op 除外)

cluster c の **全** (layer, feature) 同時に hook 登録 → 該当発火位置を一括書換え (cluster-level real intervention)。各 H につき forward を再実行。

### 3.4 Cluster-level prompts

per cluster:
- **D_probe sample**: 50 prompts。`primary_category(c) = argmax_i mean(c_f[i] over members)` → 該当 cat の positive+paraphrase prompts から `random.Random(42)` で 50 sample
- **D_utility sample**: 48 prompts (8 per subset × 6 subsets)、seed=42
- **baseline forward**: hook なしで 1 回実行 (per cluster の参照点として保存)

per intervention H × per (D_probe / D_utility) で測定:
- **D_probe**: ΔlogP(target answer) = logP_intervened − logP_baseline
- **D_utility**: Δlog_perplexity = log(ppl_intervened / ppl_baseline)
- **D_utility**: greedy 32-token generation → generation_collapse_rate 判定

### 3.5 generation_collapse_rate 定義 (spec 未定義のため自定義)

D_utility prompt p について以下の少なくとも 1 つを満たす場合 "collapsed":

1. **length_anomaly**: `|gen_token_count_int - gen_token_count_base| / max(gen_token_count_base, 1) > 0.5`
2. **repetition_loop**: 任意の 4-gram が intervention 出力で 5 回以上繰り返し
3. **empty_or_short**: `gen_token_count_int < 5`
4. **format_break**: baseline 出力に存在する punctuation (`. , : ; ? ! \n`) のうち 50% 以上が intervention 出力で完全消失

`generation_collapse_rate(c, H) = (collapsed prompts in cluster c, intervention H) / 48`

baseline = no-hook greedy generation 32 token (do_sample=False, temperature=0.0)

### 3.6 5-class labeling rules

cluster-level metrics (across H1-H7 集約):

```
te_max(c)         = max_H ΔlogP_target(c, H)        # cluster の causal "強さ"
nte_max(c)        = max_H mean_{i ≠ primary} ΔlogP_target_per_cat_i(c, H)
ud_max(c)         = max_H mean Δlog_perplexity(c, H over D_utility)
ud_var(c)         = var_H mean Δlog_perplexity(c, H over D_utility)
gen_collapse_max(c) = max_H generation_collapse_rate(c, H)
```

per-cluster percentiles を **本実験で計算した cluster 集合内** で計算:
- p50, p70, p80 over { metric_value(c) for c in valid clusters }

per-cluster classification (precedence 順、最初に該当した label を採用):

| 順位 | Label | 条件 |
|---|---|---|
| 1 | **KNOWLEDGE-ASSOCIATED** | te_max(c) ≥ p80(te_max) **AND** nte_max(c) ≤ p50(nte_max) **AND** ud_max(c) ≤ p50(ud_max) |
| 2 | **ENTANGLED** | te_max(c) ≥ p80(te_max) **AND** nte_max(c) ≥ p70(nte_max) |
| 3 | **INFRASTRUCTURE** | ud_max(c) ≥ p80(ud_max) **OR** gen_collapse_max(c) ≥ 0.30 |
| 4 | **UNSTABLE** | ud_var(c) ≥ p80(ud_var) |
| 5 | **UNKNOWN** | HDBSCAN noise (cluster_id = -1) **OR** 上記いずれにも該当せず |

threshold (p80, p70, p50, 0.30) は **実験前に固定**。post-hoc で緩和・厳格化禁止。

### 3.7 Acceptance gates

| ID | テスト | 閾値 | 含意 |
|---|---|---|---|
| **G1** | HDBSCAN valid cluster 数 (cluster_id ≥ 0) | ≥ 5 | clustering が trivial でない |
| **G2** | labeled feature coverage = (UNKNOWN 以外 cluster の member 数) / 10,311 | ≥ 0.60 | taxonomy が大半をカバー |
| **G3** | KNOWLEDGE-ASSOCIATED cluster 数 | ≥ 1 | 主目的 class が発見される |
| **G4** | label discrimination: median(te_max) of KNOWLEDGE-ASSOCIATED clusters > median(te_max) of UNKNOWN clusters | True | label が意味を持つ |
| **G5** | per-cluster heavy-vs-light correlation: cluster 内平均 te_max (heavy, H4=ρ=0.5 を採用) と cluster 内平均 c_f[primary_category] (light, Phase 3 ρ=0.5) の Pearson | ≥ 0.7 | light → heavy の transferability (light での候補絞り込みが妥当だった証拠) |

**全 G1-G5 通過時のみ** v001 Phase 4 結果を hash_log に "ALLOW" として frozen。1 つでも失敗 → `fallback_reason` に gate ID 記録 + v001.1 別 pre-reg を切る。

### 3.8 採否マトリクス

| 状態 | 判定 | v001.1 で必要なこと |
|---|---|---|
| G1-G5 全通過 | **ALLOW** | 通常 v001 完了、Phase 5 (cross-model 比較等) 検討 |
| G1 のみ失敗 | **FREEZE_G1** | UMAP 次元削減 + HDBSCAN 再試行 を v001.1 で |
| G2 のみ失敗 | **FREEZE_G2** | threshold 緩和を pre-reg せず → v001.1 で再 commit |
| G3 のみ失敗 | **FREEZE_G3** | 候補規則 (Phase 2) 見直し or threshold 過剰 → v001.1 |
| G4 失敗 | **FREEZE_G4** | label rules 見直し → v001.1 |
| G5 失敗 (light/heavy 相関弱い) | **FREEZE_G5** | linear approx の Phase 4 適用範囲再評価、Phase 3 v001.2 検討 |

## 4. Cherry picking 回避 commitments

1. **HDBSCAN params (min_cluster_size=50 等) の事後変更禁止**: cluster 数 < 5 でも変更しない、v001.1 で別 pre-reg
2. **λ 重み (1.0 / 2.0 / 2.0) の事後変更禁止**: spec 由来の値のみ使用
3. **5-class threshold (p80, p70, p50, 0.30) の事後変更禁止**
4. **7 種 intervention 全結果報告**: H1-H7 個別に削除しない、最良 1 つを後付け選定しない
5. **失敗 gate を成功として再解釈しない**: 例「G3 通らなかったが ENTANGLED が多いから OK」のような事後緩和禁止
6. **追加 H 値の事後追加禁止**: 結果見て新 ρ を追加しない
7. **z_f 構成変更禁止**: 結果悪化時に s_f / c_f / u_f の重み変更禁止

## 5. 報告コミット

実験完了後の public artifact:

- `data/profiling/v001/phase3_causal_light_pythia_160m_rho050_v2.parquet` (Step 4.1, 10,311 行 × 35 列、gitignored、hash_log audit)
- `data/profiling/v001/phase4_feature_vectors.parquet` (Step 4.2)
- `data/profiling/v001/phase4_cluster_assignment.parquet` (Step 4.3)
- `data/profiling/v001/phase4_top_activating_examples.jsonl` (Step 4.4)
- `data/profiling/v001/phase4_cluster_heavy_profile.parquet` (Step 4.5)
- `data/profiling/v001/phase4_cluster_labels.parquet` (Step 4.6)
- `data/profiling/v001/phase4_decision.json` (Step 4.7, G1-G5 + ALLOW/FREEZE)
- `configs/hash_log.json` 7 new entries
- per-cluster summary (label 分布、cluster size 分布、heavy intervention magnitude)
- 失敗時は failure 構造を明記し v001.1 pre-registration へ link

## 6. Falsifiability

| Falsifier | 検出条件 | 含意 |
|---|---|---|
| **F1** | G1 失敗 (cluster < 5) | z_f 14-dim 表現空間で意味のある cluster が形成されない → feature 表現再設計が必要 |
| **F2** | G3 失敗 (KNOWLEDGE 0 個) | "KNOWLEDGE-ASSOCIATED" feature 概念が本 setup で実体化されない → Phase 2 候補規則 / threshold 見直し |
| **F3** | G4 失敗 (label 識別力なし) | 5-class taxonomy 自体が識別性を持たない → label 規則の根本見直し |
| **F4** | G5 失敗 (light/heavy 弱相関) | attribution patching を candidate 絞りに使うこと自体の妥当性が揺らぐ → Phase 3 IG 切替検討 |

## 7. 実装 commitments

1. 本文書を git commit (時系列証跡確定)
2. `experiments/phase3_reaggregate_pythia_160m.py` 新規作成: ρ=0.5 の attribution を per-subset UD + per-cat NTE で再集約
3. `experiments/phase4_cluster_pythia_160m.py` 新規作成: z_f 構築 + HDBSCAN + top_activating + heavy intervention
4. `experiments/phase4_label_pythia_160m.py` 新規作成: 5-class label 付与
5. `experiments/phase4_decide.py` 新規作成: G1-G5 計算 + decision.json 出力
6. `pyproject.toml` に hdbscan 追加 (`uv add hdbscan`)
7. ジョブ spec 3 つ作成、scheduler 経由で実行 (Step 4.1 → 4.2-5 → 4.6-7 の順序依存)
8. decision.json 出力 → 採否判定を本文書 §3.8 に従い決定 → report

**この pre-registration は本 commit 時点で凍結される。以降の変更は v001.1 / v001.2 として別文書を切る。**
