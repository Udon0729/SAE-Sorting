# Pre-registration: Controlled LM v001 — validity track for SAE separability scan

**Pre-registered**: 2026-05-03 (UTC)
**Author**: K. Munaoka with Claude Code
**Sister track**: Pythia v001.1 (`docs/preregistrations/phase4_v001_1_measurement.md`)

## 1. 背景

Pythia v001.1 (`docs/preregistrations/phase4_v001_1_measurement.md`) は Pythia-160m 上の SAE feature の causal-risk geometry を測定し、ranking framework の validity を gate G1, G3, G4, G5 で判定する。しかし Pythia v001.1 単体では「方法に問題」と「Pythia には separable feature が存在しない」を区別できない。

本 controlled LM track は別系統の validity 検証として並走し、ground truth が既知の controlled LM 上で同 separability scan を適用、ranking が ground truth factor task に対応するかを G6 で測定する。

**本 track は Pythia v001.1 の gate を block しない**。Pythia accept/freeze 判定は Pythia v001.1 の G1, G3, G4, G5 のみ。controlled LM G6 結果は cross-track decision matrix (Pythia v001.1 §3.10) で Pythia 結果の解釈に使う。

設計上の重要な選択:

- architecture に答えを埋め込まない: 普通の decoder-only LM (Pythia-160m configuration) を使い、controlled LM 用に別ヘッド / module 化はしない。検証したいのは「通常の decoder LM に埋め込まれた factor structure を sorting が回収できるか」であり、architecture を人為的にすると Pythia/Qwen への transferability が失われる。
- tag prefix shortcut を使わない: `[GEO]<fact>` のような control token は SAE feature が tag detector になる shortcut を作る。controlled LM の corpus は通常テキストのみで構成し、factor structure は corpus design のみで実現する。
- ground truth は **feature ID レベルでなく factor task レベル**: SAE feature は訓練後にどう分かれるか不明なので、feature 単位の真ラベルを作る発想を避ける。代わりに「factor task の ground-truth causal profile を unit が回収するか」で評価する。

## 2. v001 scope (確定事項)

| 項目 | 確定 |
|---|---|
| controlled LM architecture | Pythia-160m configuration (12 layer, 768 dim, GPT-NeoX tokenizer 流用) |
| 訓練 corpus | D_synth_v001 (`data/synthetic_kg/v001/corpus/train.txt`) + 補助 `D_synth_v001_1_factor_manifest` + The Pile sample (utility 維持) |
| LM training | scratch from random init |
| SAE | 同 TopK 32k 設計 (Pythia SAE と同 setup)、controlled LM 上で再訓練 |
| evaluation | Pythia v001.1 と同 separability scan + factor task ground truth recall |
| sister gate | controlled LM G6 のみ。Pythia v001.1 gate には混ぜない |
| corpus 規模 | D_synth_v001 (現 14,840 facts) + factor manifest 補助 + The Pile sample (~1B token) |

## 3. Pre-committed design choices

### 3.1 D_synth_v001 + factor manifest 構成

D_synth_v001 (`idea/dataset.md` §3 で設計済み、`data/synthetic_kg/v001/`) は以下を既に持つ:

- C1-C7 category / relation structure
- entity / relation / compositional entanglement annotations (`entanglement_annotations.jsonl` schema は §3.5.1)
- test split: `test_unseen_template`, `test_heldout_entity`, `test_compositional`, `test_contrast`
- utility prompts (`utility/utility_prompts.jsonl`)
- annotation counts: entity 200 / relation 50 / compositional 30 (small規模)

**不足分** (本 v001 で補助生成、`data/synthetic_kg/v001/factor_manifest/` に配置):

| 出力 | 内容 |
|---|---|
| `annotation_to_qa_linkage.parquet` | annotation_id ↔ fact_id ↔ qa_id の linkage table (entanglement_annotations.jsonl の `prompt_ids` が空のため) |
| `factor_task_manifest.parquet` | factor task ID × (positive prompt_ids, negative prompt_ids, held-out prompt_ids) の mapping |
| `compositional_chain_manifest.parquet` | compositional chain ID × (chain entities, chain categories, chain prompt_ids) |
| `corpus_split_manifest.parquet` | qa_id × `controlled_split_role ∈ {train, fit, validate, generalize}` の割当 |

`factor_task_manifest.parquet` の生成 rule (本 pre-reg で固定):

- GEO-specific factor task: category == "geography" の各 relation について、positive = 該当 relation の qa_id 全部、negative = 同 entity を含む他 cat qa_id、held-out = `test_heldout_entity` の同 relation 分
- relation-specific factor task: 各 (category, relation) 組合せについて、positive = 該当 (cat, rel) qa_id、negative = 同 cat 別 relation qa_id、held-out = `test_unseen_template` の同 (cat, rel) 分
- entangled factor task: entanglement_annotations の各 annotation_id について、positive = annotation の `entities_involved` を含む全 qa_id、held-out = `test_compositional` の対応分
- utility-risk factor task: `utility/utility_prompts.jsonl` 全件 (subset 分けは Pythia v001.1 の D_utility_v001 と同様 U1-U6)

### 3.2 Factor task definition (ground truth profile)

各 factor task について、unit が満たすべき **ground truth causal profile** を pre-commit:

| Factor task | Ground truth causal profile (held-out split で評価) |
|---|---|
| GEO-specific | unit の `primary_cat` == "geography" AND CTE < CTE 中央値 (全 unit 集合内) AND held-out geography prompts での TE が同 unit の TE 全体中央値以上 |
| relation-specific | unit の primary relation TE が同 cat 内の他 relation TE の最大値の 2 倍以上 AND CTE < CTE 中央値 |
| entangled | unit が 2 cat 以上で TE 上位 25% (同 cat 内 ranking) AND 設計済み entanglement annotation の `categories_involved` と top-2 cat が一致 |
| utility-risk | unit の UD が top 5% (全 unit 集合内) AND TE が中央値以下 |

各 factor task の閾値 (本 pre-reg で固定):
- TE 上位: 25% / 中央値 (factor task ごとに記載)
- CTE 下位: 中央値
- UD 上位: top 5%
- relation specificity 比: 2 倍

### 3.3 controlled LM 訓練 spec

| 項目 | 値 |
|---|---|
| architecture | Pythia-160m configuration (12 layer, 768 dim, 12 head, 2048 ctx) |
| tokenizer | EleutherAI/gpt-neox-20b (Pythia 系流用) |
| init | scratch (random、seed=42) |
| optimizer | AdamW, lr=6e-4, β=(0.9, 0.95), weight_decay=0.1 |
| schedule | linear warmup 1% → cosine decay → 10% min |
| batch_size | 256 sequences × 2048 tokens = 524,288 tokens / step |
| total tokens | 2B tokens (≈ 4000 steps) |
| corpus mix | 70% The Pile sample + 30% D_synth_v001 corpus (混合は token-level interleave、D_synth が under-represented にならない upsample で 30% 確保) |
| precision | bf16 mixed |
| seed | 42 (data shuffling, model init, dropout) |

訓練 wall: 約 半日 (RTX PRO 6000 1 枚)。本 pre-reg は controlled LM 訓練の **1 回 clean run** のみで評価する (再訓練禁止、結果見ずに固定)。

### 3.4 SAE 訓練 spec

| 項目 | 値 |
|---|---|
| architecture | TopK SAE, k=32 (Pythia SAE と同 setup) |
| width | 32,768 latent (32k = Pythia SAE と同) |
| target | controlled LM の各 residual stream layer 出力 (12 layer 全て訓練、解析対象 layer は Pythia 解析と同層) |
| training corpus | controlled LM 訓練 corpus と同 (mix も同) |
| training tokens | 100M tokens (Pythia SAE と同オーダー) |
| optimizer | Adam, lr=1e-4 |
| batch_size | 4096 activation vectors / step |
| seed | 42 |

SAE 訓練 wall: 約 数時間 / layer × 12 layer ≈ 1-2 日 (順次)。

### 3.5 Separability scan application

Pythia v001.1 と同 pipeline を controlled LM 上に適用:

- 7-metric measurement (TE, CTE, UD, Stability, Specificity, Editability, Confidence)
- damage = baseline_logP − intervened_logP の符号統一
- NaN handling: c_f 全 cat non-NaN limit (controlled LM では cat = controlled corpus の C1-C7 のうち実体化されたもの、C8 behavioral は除外)
- D_probe / D_utility 相当: factor_task_manifest.parquet の positive / held-out + utility_prompts.jsonl
- attribution patching ρ=0.5 (Pythia と同設定)
- Stratified heavy validation: Pythia と同 8 strata × 30 = 240 feature (controlled LM での light scan 後)
- Top-k group: 同 k ∈ {2, 3, 5}
- Compression layer: 同 4 手法 × 3 粒度

### 3.6 G6 acceptance gate

#### 3.6.1 Task-specific Specificity ranking

各 factor task について、その task に固有の prompt 集合を使って **task-specific** TE / CTE / Specificity を計算する。global Specificity ranking を全 task に流用しない (流用すると G6 は意味を失う):

```
TE_task(unit)          = max_H  mean_{p ∈ positive_prompt_ids(task)} damage_target(unit, p, H)
CTE_task(unit)         = max_H  mean_{p ∈ negative_prompt_ids(task)} damage_target(unit, p, H)
Specificity_task(unit) = TE_task(unit) - CTE_task(unit)
```

utility-risk task は §3.6.3 で別ルール。

#### 3.6.2 Recall 計算 (dynamic N_tasks)

```
TopK_task          = task-specific Specificity_task 降順 top-50 unit (task ごと独立に計算)
recall_factor_task = |{task : ∃ unit ∈ TopK_task. profile_match(unit, task)}| / N_tasks
```

`N_tasks` は `factor_task_manifest.parquet` 生成後の **availability filter 通過後の実 task 数** で確定する (hard-code しない、§3.6.5 参照)。

#### 3.6.3 Profile match 判定 (per factor task)

- **GEO-specific match**: unit の `primary_cat` (global TE で判定) == "geography" AND `CTE_task(unit) < CTE_task の中央値 (同 task 内全 unit)` AND `held-out GEO prompts での TE_task(unit) > TE_task の中央値`
- **relation-specific match**: unit の `argmax_relation TE_task` が target relation AND `TE_task(target relation) > 2 × max_{other rel in same cat} TE_task` AND `CTE_task < CTE_task 中央値`
- **entangled match**: unit の `top-2 cat by TE_task` が annotation の `categories_involved` と集合一致 AND `各 cat の TE_task > 該当 cat 内 TE_task の p75`
- **utility-risk match**: unit の `UD > UD_p95(全 unit, global)` AND `TE < TE_median(全 unit, global)` (utility-risk のみ global TE/UD を使う、task-specific は意味を持たない)

#### 3.6.4 G6 閾値

```
pass_count = ceil(0.7 × N_tasks)
G6 PASS ⇔ (回収 task 数) ≥ pass_count
```

#### 3.6.5 Task 数の cap (availability-aware)

各 factor task カテゴリの実 task 数は `min(available, cap)` で確定する。`available` は `factor_task_manifest.parquet` 生成時に確定する実数 (D_synth v001 の現実 relation / annotation 数に基づく):

| Factor task カテゴリ | cap | available 例 (D_synth v001 small) |
|---|---|---|
| GEO-specific | (cap なし、geography の全 relation) | 1 (`located_in` のみ) |
| relation-specific | 30 (random, seed=42) | min(18, 30) = 18 |
| entangled (entity) | 30 (random, seed=42) | min(available, 30) |
| entangled (relation) | 15 (random, seed=42) | min(available, 15) |
| entangled (compositional) | 5 (random, seed=42) | min(available, 5) |
| utility-risk | 6 (U1-U6 fixed) | 6 |

`available < cap` の場合は全件採用 (random sampling 不要)、`available ≥ cap` の場合は seed=42 stratified random で cap 件選定。

正式 `N_tasks` は manifest 生成 script (`experiments/controlled_v001_factor_manifest.py`) の出力 log と `factor_task_manifest.parquet` で確定し、`data/profiling/controlled_v001/decision.json` に `n_tasks_actual`、`pass_count`、`recovered_count` を記録する。

### 3.7 Cross-track decision matrix

(Pythia v001.1 §3.10 を再掲、両 track の組合せ)

**Validity precondition**: controlled LM が D_synth qa を baseline で解けない、または SAE reconstruction が破綻している場合は、validity track は **INVALID_RUN** となり G6 を計算しない (§6.1 参照)。INVALID_RUN の場合、Pythia v001.1 は独立に read され、controlled validity は本 run では assessable でないと note 付き report する。

| Pythia v001.1 | controlled LM 結果 | 解釈 | 次アクション |
|---|---|---|---|
| ALLOW (G1, G3, G4, G5 全 PASS) | G6 PASS (≥ 0.7) | 方法 valid、Pythia には separable feature が **存在** | Phase 5: Qwen3-8B 適用、cross-model geometry 比較 |
| FREEZE (G1/G3/G4/G5 いずれか FAIL) | G6 PASS | 方法 valid、Pythia には separable feature が **薄い** (限定的 negative) | controlled LM 結果を主結果として report、Pythia は negative finding として scope 限定 (Pythia-160m + 現 SAE + 現 D_probe + 現 intervention) |
| ALLOW | G6 FAIL (< 0.7) | 矛盾、原因究明要 | controlled LM 訓練 / SAE 訓練 / corpus 設計の問題を疑う、追加診断 |
| FREEZE | G6 FAIL | 方法に問題 | separability scan / metric / pipeline 見直し、v001.2 redesign |
| ALLOW | INVALID_RUN | controlled validity 未確定 | Pythia 結果を主とし、controlled v002 (規模拡大 / 訓練改善) で validity 再判定 |
| FREEZE | INVALID_RUN | Pythia FREEZE の原因が方法 vs モデルか判定不能 | controlled v002 で validity 再判定後に Pythia 結果の解釈を確定 |

## 4. Cherry-picking 回避 commitments

1. **controlled LM 訓練を結果見て再開しない** (1 回 clean run)
2. **SAE 訓練を結果見て再開しない**
3. **factor task definition / 閾値を pre-commit 後変更しない**
4. **G6 閾値 (0.7) を結果見て緩和しない**
5. **factor task cap (relation 30 / entangled entity 30 / entangled relation 15 / entangled compositional 5) と availability filter `min(available, cap)` を seed=42 stratified random で固定、結果見て差替えない。`N_tasks` は manifest 生成後の動的確定値を採用、結果見て手動修正しない**
6. **profile match 判定 rule (§3.6) を結果見て緩和しない**
7. **corpus mix 比率 (70% Pile / 30% D_synth) を結果見て変更しない**
8. **訓練 token 数 (controlled LM 2B, SAE 100M) を結果見て増減しない**

## 5. 報告コミット

実験完了後の public artifact:

- `data/synthetic_kg/v001/factor_manifest/annotation_to_qa_linkage.parquet`
- `data/synthetic_kg/v001/factor_manifest/factor_task_manifest.parquet`
- `data/synthetic_kg/v001/factor_manifest/compositional_chain_manifest.parquet`
- `data/synthetic_kg/v001/factor_manifest/corpus_split_manifest.parquet`
- `data/lm/controlled_v001/` (controlled LM checkpoint)
- `data/sae/controlled_v001/` (controlled SAE per layer)
- `data/profiling/controlled_v001/phase3_causal_light_controlled_rho050_v3.parquet`
- `data/profiling/controlled_v001/phase4_v001_1_unit_table.parquet`
- `data/profiling/controlled_v001/factor_task_recall.parquet` (per factor task × match status)
- `data/profiling/controlled_v001/decision.json` (G6 + PASS/FAIL)
- `configs/hash_log.json` 7+ new entries

## 6. Validity preconditions and Falsifiability

### 6.1 Validity preconditions (INVALID_RUN triggers)

以下のいずれかが満たされる場合、controlled LM track は **INVALID_RUN** 扱いとし、G6 を計算せず `decision.json` に `status: "INVALID_RUN"` と記録する。これは方法の falsification ではなく、validity track の **実験成立条件未満** の認定である:

| Trigger | 検出条件 | 対応 |
|---|---|---|
| **P1 (LM)** | controlled LM が D_synth `test_seen.jsonl` で baseline accuracy < 50% (next-token argmax で正答率) | controlled LM 訓練が成立せず → corpus mix 見直し / 訓練 token 増を controlled v002 で対応 |
| **P2 (SAE)** | SAE reconstruction MSE が Pythia SAE on Pythia-160m の同 layer reconstruction MSE の 2 倍以上、または explained variance < 0.6 | SAE 訓練が成立せず → SAE training tokens / lr / TopK 値見直しを controlled v002 で対応 |

INVALID_RUN の場合、§3.7 cross-track decision matrix の INVALID_RUN 行を適用する (Pythia 結果のみで limited interpretation)。

### 6.2 Falsifiers (G6 計算後の解釈)

| Falsifier | 検出条件 | 含意 |
|---|---|---|
| **F8** | G6 < 0.3 (大幅に下回る) AND Pythia v001.1 ALLOW | controlled LM の factor structure が SAE 内で実体化されていない → corpus design / annotation 規模 (v001 small) の不足、controlled v002 で main 規模 (50k facts) で再試行 |
| **F9** | G6 ≥ 0.7 AND Pythia v001.1 FREEZE | 方法は valid、Pythia には薄い → 限定的 negative finding として scope 限定 report |
| **F10** | G6 ≥ 0.7 AND Pythia v001.1 ALLOW | 主仮説 supported、Phase 5 進行可 |

## 7. 実装 commitments

1. 本文書を git commit (時系列証跡確定)
2. `experiments/controlled_v001_factor_manifest.py` 新規作成: D_synth_v001 から annotation linkage / factor task manifest / chain manifest / split manifest を生成
3. `experiments/controlled_v001_train_lm.py` 新規作成: Pythia-160m architecture scratch 訓練
4. `experiments/controlled_v001_train_sae.py` 新規作成: TopK SAE 訓練 (layer ごと sequential)
5. `experiments/controlled_v001_phase3_light.py` 新規作成: Pythia v001.1 の Phase 3 v3 を controlled LM に適用
6. `experiments/controlled_v001_phase4_separability.py` 新規作成: Pythia v001.1 の separability scan を controlled LM に適用 (Pythia 用と shared library 化)
7. `experiments/controlled_v001_decide.py` 新規作成: G6 計算 + decision.json 出力
8. ジョブ spec 6 つ作成 (factor manifest → LM 訓練 → SAE 訓練 → light → separability → decide)
9. `pyproject.toml` に追加 (Pythia v001.1 と共通: leidenalg, igraph, POT)
10. decision.json 出力 → 採否判定を本文書 §3.7 cross-track decision matrix に従い読む

**この pre-registration は本 commit 時点で凍結される。以降の変更は controlled v002 として別文書を切る。**
