# Pre-registration: Phase 4 v001.1 — causal-risk measurement & per-feature ranking (separability scan)

**Pre-registered**: 2026-05-03 (UTC)
**Author**: K. Munaoka with Claude Code
**Predecessor**: Phase 4 v001 (`docs/preregistrations/phase4_v001.md`) → `FREEZE_G1_G2_G3_G4_FAIL` (`data/profiling/v001/phase4_decision.json`)
**Sister track**: Controlled LM validity (`docs/preregistrations/controlled_v001_validity.md`)

## 1. 背景

Phase 4 v001 は **5-class taxonomy 自動発見** を主目的とし、HDBSCAN clustering + cluster-level heavy intervention + percentile-based label assignment で構成した。結果は 5 gates 中 4 失敗 (G1: clusters=4 < 5、G2: coverage=0.157、G3: KNOWLEDGE=0、G4: undefined median; G5 のみ pass)。

事後分析で次の方法論的問題を識別した:

1. **Circular labeling**: 「KNOWLEDGE = (TE > p80, NTE < p50, UD < p50)」のような閾値定義は tautology である。閾値を引いて該当物に名前を付けただけで、experimental discovery ではない。
2. **Model-dependent class count**: Pythia / Qwen3 / controlled LM で同じ 5 種類が同じ粒度で出る根拠はない。pre-committing 5 classes for any model は方法論的に unjustified。
3. **Cluster aggregation hides feature operability**: cluster 3 (1615 features) で te_max=0.12 が観察されたが、これは 1615 feature 同時介入の総和効果であり、「単一 / 小 group の操作単位として specificity が高い」ことを示さない。
4. **NaN-driven degenerate geometry**: c_f 4 cat に NaN を含む feature が 94% に達した。`fillna(0)` で clustering 可能にしたが、これは「fired but no measurable effect = 0」と「not fired in this cat = structurally undefined」を conflate しており、representation 空間を degenerate にする。
5. **Heavy schema missing cross-cat**: heavy intervention で primary cat の TE のみ測定し、他 cat の damage を測らなかったため、ENTANGLED 判定が原理的に不可能だった。
6. **No held-out validation**: ranking が in-fit prompts でのみ計算され、generalization が確認されていない。
7. **No external ground truth**: Pythia 結果が「方法の bug」か「Pythia にそういう feature がない」か区別できない。

**v001.1 はこれらを構造的に解決するため、目的を以下に再定義する**:

> SAE feature / small feature group に、target causal effect と non-target / utility damage を分離できる操作単位が、どの程度・どの条件で存在するかを測る。

これは **separability scan + per-unit ranking framework** であり、taxonomy verification ではない。5-class label は report 末尾の diagnostic vocabulary に降格する (本 pre-reg では pre-commit せず success criterion でない)。

**v001.1 の成功は、特定ラベルの発見ではなく、causal-risk measurement が再現可能で、light proxy が heavy validation に transfer し、ranking が held-out split で actionability を持つことによって定義する。**

Pythia 結果の解釈は、別 pre-reg `controlled_v001_validity.md` で並走する controlled LM ground truth により裏打ちされる。本 pre-reg (Pythia v001.1) は controlled LM の成否でブロックされず、Pythia 上の measurement validity と ranking actionability のみを判定する。

## 2. v001.1 scope (確定事項)

| 項目 | v001.1 | v002 deferred |
|---|---|---|
| model | Pythia-160m-deduped @ step143000 | Qwen3-8B v001.1 |
| SAE | EleutherAI/sae-pythia-160m-deduped-32k | – |
| 入力 (light) | Phase 3 v001.2 出力 (per-prompt logP retention 拡張、ρ=0.5) | – |
| categories | 4 (D_probe v001 native: person_attribute / geography / organization / occupation) | 8 cat 拡張 |
| utility subsets | 6 (D_utility v001 native: U1-U6) | – |
| heavy interventions | 7 active (zero, mean, scale_down ρ ∈ {0.25, 0.5, 0.75}, negative_scale ρ ∈ {0.25, 0.5}) | – |
| heavy validation scope | 8 strata × 30 = 240 single feature + top-k group (k ∈ {2, 3, 5}) for top-50 anchor | full 10,311 candidate heavy |
| split usage | fit (selection) / validate (gate) / generalize (report) — held-out entity/instance のみ claim | relation held-out (D_probe v001.2 で別 split 生成後に claim) |
| 5-class taxonomy | report-only diagnostic vocabulary、**success criterion でない** | – |

## 3. Pre-committed design choices

### 3.1 Indicator unification

すべての damage 指標を以下に統一:

```
damage(unit, target) = logP_baseline(target) - logP_intervened(target)
```

- 正値 = intervention が target を hurt する
- abs() / raw / 符号反転表記の混在を全面禁止
- TE, CTE, UD すべてこの定義で計算

### 3.2 Measurement space (7 metrics per unit)

各 unit (single feature / small group / cluster prototype) について 7 指標を測定:

| 指標 | 定義 | 単位 |
|---|---|---|
| TargetEffect (TE) | `max_c damage_target_cat(unit, c)`、c ∈ 4 cat | nat |
| CrossTargetEffect (CTE) | `max_{c' ≠ argmax_c TE} damage_target_cat(unit, c')` | nat |
| UtilityDamage (UD) | `z(max_subset damage_utility(unit, subset))` と `z(max_subset gen_collapse_rate(unit, subset))` の max (各 z は全 unit 集合内の z-score) | dimensionless (合成) |
| Stability | `-var_H damage_target_primary(unit, H)`、H ∈ {scale_down ρ ∈ {.25, .5, .75}, negative ρ ∈ {.25, .5}} (5 H) | nat² (負号により低 variance = 高 stability) |
| Specificity | `TE - CTE` | nat |
| Editability | `TE / (1 + α · UD_raw)`、α ∈ {0.5, 1, 2} を grid 報告、主 ranking には α = 1 を採用、`UD_raw` は z-score 前の `max_subset damage_utility` | nat (相対) |
| Confidence | `1 / median(CI半幅)`、prompt-level bootstrap (1000 resample, seed=42) で TE と UD の median CI 半幅を計算し、両者の調和平均の逆数 | dimensionless |

`primary_cat(unit) = argmax_c damage_target_cat(unit, c)` で定義。

### 3.3 Phase 3 v001.2 拡張 (新出力)

新規出力 `data/profiling/v001/phase3_causal_light_pythia_160m_rho050_v3.parquet`:

- 既存 v2 schema (35 列) を継承
- 追加: per-(layer, feature, prompt_id) level `logP_baseline`, `logP_intervened` を retain (long format、約 10,311 × 全 prompt 行)
- これにより heavy なしで bootstrap CI を計算可能、再 forward 不要

heavy 出力 `data/profiling/v001/phase4_v001_1_heavy_per_unit.parquet`:

- per (unit_type, unit_id, intervention_H, split_role, prompt_id) で `baseline_logP_target`, `intervened_logP_target`, `baseline_logP_target_per_cat[c]`, `intervened_logP_target_per_cat[c]`, `gen_token_count_baseline`, `gen_token_count_intervened`, `gen_4gram_max_repeat`, `gen_punct_baseline`, `gen_punct_intervened` を全て保存
- 後続の bootstrap / median / max 集計は post-hoc でも、aggregation rule は本 pre-reg で固定 (§3.2)

### 3.4 NaN handling

- 解析対象: c_f 4 cat (`delta_m_target_per_cat_person_attribute` ... `delta_m_target_per_cat_occupation`) のすべてが non-NaN な feature
- 除外後の n を report
- n < 500 の場合は fallback として missingness mask 列 (4 binary) を分離追加し、解析対象を c_f any non-NaN に拡張 (本 pre-reg で fallback rule を pre-commit、結果見て選ばない)
- NaN の意味論的解釈は v001.1 では「c_f 4 cat 全 non-NaN feature だけ ranking 対象」と保守的に扱う

### 3.5 Stratified heavy validation (240 features)

8 strata × 30 features = 240 features を heavy 介入で測定。stratum 定義 (priority 順、上位 stratum に既割り当ての feature は下位重複させない):

| Stratum | 定義 | source |
|---|---|---|
| S1 | light TE 上位 30 (Phase 3 v3 fit split で `delta_m_target` 絶対値) | Phase 3 v3 |
| S2 | light Specificity 上位 30 (light c_f primary − light c_f 2nd max) | Phase 3 v3 |
| S3 | light UD 下位 30 (`delta_m_utility` 絶対値小) | Phase 3 v3 |
| S4 | light Pareto frontier 候補 (TE 大かつ UD 小) 上位 30 (Pareto domination filter 後 TE 降順) | Phase 3 v3 |
| S5 | HDBSCAN noise pool (Phase 4 v001 cluster_id == −1) から random 30 | Phase 4 v001 |
| S6 | candidate_type == broad (Phase 2) から random 30 | Phase 2 |
| S7 | candidate_type == category_selective から random 30 | Phase 2 |
| S8 | random control: 全 candidate (NaN handling 通過後) から uniform random 30 | Phase 3 v3 |

random selection は `np.random.default_rng(42)` で stratified sampling、selection script (`experiments/phase4_v001_1_stratify.py`) を git commit。

各 stratum × feature について、heavy intervention 7 H × per-cat D_probe 50 prompts × **4 cat 全測定 (cross-cat)** + per-subset D_utility 8 prompts × 6 subset を fit / validate / generalize 全 split で測定。

estimated forwards: 240 × 7 × (50 × 4 + 8 × 6) = 240 × 7 × 248 = 416,640 forwards。Pythia-160m fp32 で約 5–7 hours wall (RTX PRO 6000)。

### 3.6 Top-k small group

stratified heavy validation 完了後、240 single feature のうち TE 降順 **top 50** を anchor として small group 評価:

- anchor: stratified heavy 240 feature のうち TE (validate split) 降順 top 50
- neighbor pool: c_f 4 cat 全 non-NaN かつ fit split light profile を持つ 全 candidates
- distance: standardized fit-split light causal-risk vector (TE_light, CTE_light, UD_light, Stability_light の 4 dim z-score) の cosine distance
- group: anchor + nearest (k − 1) features、k ∈ {2, 3, 5}
- multi-layer OK (同 layer 制限なし、編集単位として multi-layer group を許容)
- **co-firing pair は本 v001.1 では不採用** (Phase 1 firing pattern overlap は別仮説、v001.2 へ defer)
- 各 group について heavy intervention 7 H × cross-cat D_probe + utility を validate / generalize split で測定 (anchor の fit split 結果は selection に使い、group 評価は held-out のみ)
- group 単位の TE / CTE / UD / Stability / Specificity / Editability / Confidence を per-unit table に追記
- group 内の primary category composition (4 cat の TE 比率) と layer composition (12 layer の分布) を report

estimated forwards: 50 anchor × 3 group sizes × 7 H × 248 = 260,400 forwards (validate + generalize split のみ)。約 3–4 hours wall。

### 3.7 Compression layer (subordinate)

per-unit table 完成後、説明・編集単位として圧縮表現を構築 (主 success criterion でない、副次 layer):

4 手法 × 3 粒度:
- HDBSCAN: density baseline (`min_cluster_size ∈ {10, 30, 100}`)
- Leiden: kNN graph (k = 15) → resolution ∈ {0.5, 1.0, 2.0}
- agglomerative: ward linkage、`n_clusters ∈ {10, 50, 200}`
- UOT (unbalanced optimal transport): soft assignment、`n_prototypes ∈ {10, 50, 200}` (entropic regularization ε=0.1, marginal relaxation τ=1.0)

入力空間: 7 metrics の per-axis z-score (10,311 行のうち NaN handling 通過分)

評価指標:
- **hard comparison**: 4 手法間 Top-1 assignment Jaccard (UOT は max-mass argmax で hard 化)
- **soft comparison (UOT)**: transport-plan cosine、entropy、mass concentration
- **quality**: within-unit Specificity variance (低いほど良い、prototype 内 unit が specificity 的に均質)
- **stability**: bootstrap (1000 resample, seed=42) co-assignment probability
- **editability**: cluster / prototype weighted TE, CTE, UD (member 重み平均)

UOT は **feature → edit prototype の重み行列** として評価する (clustering 手法としてだけでなく、downstream editing 用 transport plan として report)。

### 3.8 Acceptance gates

| ID | テスト | 閾値 | 性質 |
|---|---|---|---|
| **G1** | feature-level light TE ↔ heavy TE Spearman correlation across 240 feature (validate split) | ≥ 0.7 | pipeline validity (light → heavy transferability) |
| **G2** | Specificity 分布の IQR / range / 90-10 percentile gap (240 feature, validate split) | **report only** (no pass/fail) | 分布の spread 自体が discovery 内容、spread 薄なら「このモデルでは separable feature 群は薄い」を結論 |
| **G3** | held-out (validate) feature ranking 再現: K=20/50/200 Top-K Jaccard between fit-ranking and validate-ranking + Spearman correlation + NDCG@200。**主 gate**: K=200 Jaccard ≥ 0.7 AND Spearman ≥ 0.7 (fit 240 feature と validate 240 feature の TE ranking の比較)。K=20/50 と NDCG@200 は report only | K=200 Jaccard ≥ 0.7 AND Spearman ≥ 0.7 | non-overfit ranking |
| **G4** | fit split で top-K=50 Editability features 選定 → validate split で評価。比較対象: (a) random 50 features、(b) high-TE-only baseline (fit split で TE 降順 top 50)。**主 gate (K=50)**: validate split で (i) `TE_editability の bootstrap 95% CI lower bound ≥ 0.95 × median(TE_highTE_baseline)` (non-inferiority margin 5%)、AND (ii) `UD_editability < UD_highTE_baseline` (Mann-Whitney U test, p < 0.05、Cliff's δ report、bootstrap CI 報告)。K=20/200 は report only。generalize split は別 report | (i) AND (ii) at K=50 | ranking actionability |
| **G5** | top-K=50 (TE / Specificity / Editability 各々で top-50) candidates の signal-to-noise: median(CI 半幅) / |median(value)| ≤ 0.30 (validate split, bootstrap CI) | 3 ranking それぞれで ≤ 0.30 | 測定 noise が信号未満 |

**G6 (controlled LM validity) は本 pre-reg に含めない**。`controlled_v001_validity.md` で別途判定し、Pythia v001.1 の accept/freeze 判定は G1, G3, G4, G5 のみで行う (G2 は spread を report only)。

### 3.9 Split usage

- selection: `split_role == fit` (713 prompts、約 178 / cat)
- gate: `split_role == validate` (251 prompts、約 63 / cat) — held-out entity/instance split として使う (subject mostly separated, relation overlapping per `data/probe/v001/config.yaml` の `random_with_contrast_group_integrity` 戦略)
- report: `split_role == generalize` (236 prompts、約 59 / cat) — 同様に held-out entity/instance split。validate と別々に G3, G4 を計算し、両者を separately report
- **relation-held-out split は D_probe v001 では構造上 clean でない** (relation が fit/validate/generalize 間で重複)。本 pre-reg では relation axis を claim せず、relation generalization は将来 D_probe v001.2 (`relation_held_out_generalize` 戦略) で別 commit する
- template / paraphrase held-out 評価も本 v001.1 では gate にしない (D_probe v001 構造で clean でない)

### 3.10 Decision matrix

| 状態 | 判定 | 含意 |
|---|---|---|
| G1, G3, G4, G5 全通過 | **ALLOW** | v001.1 結果を accepted measurement framework として hash_log に frozen。Phase 5 (Qwen3-8B v001.1 適用、cross-model geometry 比較) 検討可 |
| G1 のみ失敗 | **FREEZE_G1** | light approximation が heavy に transfer しない → attribution patching の Phase 4 適用範囲再評価 |
| G3 のみ失敗 | **FREEZE_G3** | ranking が overfit。NaN handling 強化 / heavy 標本数増 を v001.2 で検討 |
| G4 失敗 | **FREEZE_G4** | Editability ranking が actionable でない。Editability 定義 α grid 再評価、UD 統合方法見直し |
| G5 失敗 | **FREEZE_G5** | 測定 noise が大きい。bootstrap resample 数増、prompt 数増 を検討 |
| 複数失敗 | **FREEZE_<gate1>_<gate2>_..._FAIL** | 各 gate ごとに `fallback_reason` 記録 |

任意の Pythia v001.1 結果は controlled LM (`controlled_v001_validity.md`) の G6 結果と合わせて読む:
- Pythia ALLOW + controlled LM G6 PASS → 「方法 valid、Pythia には separable feature が **存在**」
- Pythia FREEZE + controlled LM G6 PASS → 「方法 valid、Pythia には separable feature が **薄い**」(限定的 negative)
- Pythia FREEZE + controlled LM G6 FAIL → 「方法に問題」(separability scan / metric / pipeline 見直し)
- Pythia ALLOW + controlled LM G6 FAIL → 矛盾、原因究明要 (例: controlled LM 訓練 / SAE 訓練の問題)

## 4. Cherry-picking 回避 commitments

1. **Stratum 定義 (8 strata × 30) を結果見て変更しない**: 結果が trivial でも変更せず v001.2 で別 commit
2. **Top-k group の k ∈ {2, 3, 5} を結果見て増減しない**
3. **Acceptance gate 閾値 (G1: 0.7, G3: 0.7, G4 NIM: 0.95, G4 UD: p < 0.05, G5: 0.30) を結果見て緩和しない**
4. **7 種 intervention 全結果報告、選別禁止**
5. **5-class diagnostic vocabulary を success criterion に格上げしない** (report 末尾の便宜的 vocabulary に固定)
6. **Phase 3 v001.2 出力 schema を pre-reg 起稿後に変更しない**
7. **Compression layer 4 手法 × 3 粒度を結果見て削らない**
8. **失敗 gate を成功として再解釈しない**
9. **bootstrap seed (=42)、resample 数 (=1000)、prompt-level resample 単位を結果見て変更しない**
10. **Editability α grid {0.5, 1, 2} のうち主 ranking には α=1 を採用 (結果見て切替えない)**

## 5. 報告コミット

実験完了後の public artifact:

- `data/profiling/v001/phase3_causal_light_pythia_160m_rho050_v3.parquet` (Phase 3 v001.2 拡張、per-prompt logP)
- `data/profiling/v001/phase4_v001_1_stratum_assignments.parquet` (240 feature × stratum tag)
- `data/profiling/v001/phase4_v001_1_heavy_per_unit.parquet` (heavy 詳細、per-prompt level)
- `data/profiling/v001/phase4_v001_1_unit_table.parquet` (per-unit × 7 metrics + bootstrap CI、single feature と top-k group の両方)
- `data/profiling/v001/phase4_v001_1_compression.parquet` (4 手法 × 3 粒度の partition assignment)
- `data/profiling/v001/phase4_v001_1_compression_eval.json` (Jaccard, soft-eval, quality, stability)
- `data/profiling/v001/phase4_v001_1_decision.json` (G1, G3, G4, G5 + ALLOW/FREEZE)
- `configs/hash_log.json` 8 new entries (各 artifact の sha256 + decision_status)
- summary report (per-unit table summary、ranking distribution、TE-UD geometry plot、diagnostic vocabulary report)

## 6. Falsifiability

| Falsifier | 検出条件 | 含意 |
|---|---|---|
| **F1** | G1 失敗 | light proxy が heavy validation に transfer しない → attribution patching の Phase 4 spec 全体の妥当性が揺らぐ。IG (integrated gradients) 切替検討 |
| **F2** | G3 失敗 | ranking が in-fit 限定 → measurement framework が generalize しない。標本数増 / NaN handling 強化 |
| **F3** | G4 失敗 | Editability ranking が high-TE-only baseline と比較して non-inferior な TE を保ちながら UD を下げられない → 「TE と UD は分離不能」or 「現 Editability 定義が actionable でない」 |
| **F4** | G5 失敗 | top-K candidates の bootstrap CI が信号と同程度 → 標本数 / metric noise が ranking の安定性を阻害 |
| **F5 (cross-track)** | controlled LM G6 失敗 | 方法自体に問題 → Pythia 結果は方法問題の現れであり、Pythia の separability の有無について claim できない |

## 7. 実装 commitments

1. 本文書を git commit (時系列証跡確定)
2. `experiments/phase3_v001_2_per_prompt_logp.py` 新規作成: Phase 3 ρ=0.5 を per-prompt logP 保存で再実行 (既存 attribution code 再利用、ρ=0.5 のみ、long format 出力)
3. `experiments/phase4_v001_1_stratify.py` 新規作成: 8 strata × 30 = 240 feature の stratum assignment
4. `experiments/phase4_v001_1_heavy.py` 新規作成: 240 feature × cross-cat heavy + top-50 group heavy (vectorized hook、Phase 4 v001 の vectorized pattern を再利用)
5. `experiments/phase4_v001_1_unit_table.py` 新規作成: 7 metrics 集計 + bootstrap CI
6. `experiments/phase4_v001_1_compression.py` 新規作成: 4 手法 × 3 粒度 + 評価指標
7. `experiments/phase4_v001_1_decide.py` 新規作成: G1, G3, G4, G5 計算 + decision.json 出力
8. ジョブ spec 6 つ作成 (Phase 3 v3 → stratify → heavy → unit_table → compression → decide の順、scheduler 経由で実行)
9. `pyproject.toml` に追加: leidenalg、python-igraph、POT (UOT 用)
10. decision.json 出力 → 採否判定を本文書 §3.10 に従い決定 → report

**この pre-registration は本 commit 時点で凍結される。以降の変更は v001.2 / v001.3 として別文書を切る。**
