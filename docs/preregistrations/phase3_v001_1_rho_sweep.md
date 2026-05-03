# Pre-registration: Phase 3 v001.1 — ρ sweep sensitivity analysis (v2)

**Pre-registered**: 2026-05-03 (UTC)
**Author**: K. Munaoka with Claude Code
**Supersedes**: 内部 v1 ドラフト (git commit 前に内省で破棄)。v1 は線形 attribution が ρ で proportionate になる性質を C2-C4 が無視していた致命的欠陥を含んでいた。本 v2 は real intervention ベースに修正済。

## 1. 背景 (短縮)

Phase 3 v001 (job 0054) で attribution patching scale_down ρ=0.5 → Pearson(linear, real)=0.71 / Spearman 0.93 / median rel_err 0.13。doc gate (Pearson ≥ 0.85) 未通過。

post-hoc に ρ を変えて再試行 = researcher degrees of freedom (cherry picking 同種)。**Miller et al. 2024** "Faithfulness Metrics Are Not Robust" (arXiv:2407.08734) / **Li & Janson 2024** "Optimal Ablation for Interpretability" (arXiv:2409.09951) の精神に則り、**結果を見る前に sweep ρ と採否規則を pre-commit**。

## 2. v1 内省で発見した欠陥と本 v2 の修正

線形 attribution 式 `ΔM_linear(f, t; ρ) = (ρ - 1) · val_f_t · (W_dec[f] · ∇M(z_t))` において、`(ρ - 1)` は ρ ∈ {0.25, 0.5, 0.75} 全て**負の定数**。任意 (feature, prompt) について:

- 符号は ρ で同一 → v1 C2 (TE_dom 符号一致) は trivially 100%
- per-feature rank は ρ で同一 (比例関係) → v1 C3 (Spearman) は trivially 1.0
- |TE_dom|/|NTE_other| 比は ρ に invariant (分子分母同係数 scale) → v1 C4 比率も ρ で同一

故に v1 の C2-C4 は **線形のみで自動通過**、sensitivity test として無効だった。

**v2 修正**: C2-C5 を **REAL intervention 値** または **linear-vs-real 比較** に基づくテストに変更。線形のみで trivial 満足になる規則は排除。

## 3. 実験 design (pre-committed)

### 3.1 ρ sweep
- ρ ∈ **{0.25, 0.5, 0.75}** (sweep 中央値 0.5 を v001 と一致させ連続性確保)
- 0/1 端は除外 (ρ=0 は doc spec で Phase 3 から除外、ρ=1 は no-op)
- 各 ρ で linear attribution (全 1500 prompt) + real intervention validation 実行

### 3.2 Validation pool
- n = **500** (v001 の 200 から増。Pearson r の SE は n=500 で ~0.025、3 ρ 比較の解像度に十分)
- seed=42 を 3 ρ で **共有** (同一 (feature, prompt) pair に対する linear vs real を 3 ρ で揃える)
- ρ=0.5 では v001 (n=200) と異なる pool になるため結果は v001 と独立 (二重カウントしない)

### 3.3 採否規則 (pre-committed)

| ID | テスト | 閾値 | 根拠 |
|---|---|---|---|
| **C1** | per-ρ Pearson(linear, real) | (gate なし、報告のみ) | 透明性 |
| **C2** | 同一 500 pair の **REAL TE 値**: 3 ρ 全て同符号 pair の比率 | ≥ 75% | random baseline 25% (= 2/2³) の 3 倍 |
| **C3** | **REAL TE 値** の 3 ρ ペアワイズ Spearman の最小値 | ≥ 0.80 | doc validation gate 同水準 |
| **C4** | per-ρ Pearson(linear, real) の max − min | ≤ 0.10 | linear 近似精度の ρ-robustness |
| **C5** | 少なくとも 1 つの ρ で Pearson(linear, real) | ≥ 0.85 | doc validation gate (linear achievable regime の存在性) |

### 3.4 Phase 4 入力選定 (pre-committed)
- **ρ=0.5 の linear attribution 結果を Phase 4 入力に採用**
- 根拠: (a) sweep 中央値、(b) v001 と同一値で連続性、(c) 結果を見る前の固定選択
- ensemble / Pearson-weighted 等の代替案は v001.2 で検討、v001.1 では行わない

### 3.5 採否マトリクス

| 状態 | Phase 4 進入 | v001.2 で必要なこと |
|---|---|---|
| C2-C5 全通過 | **許可**、ρ=0.5 結果採用 | 通常 Phase 4 進行 |
| C5 のみ失敗 (3 ρ 全て Pearson < 0.85) | **凍結** | IG (Integrated Gradients, Marks 2024 方式) 切替 |
| C4 失敗 (Pearson 強く ρ 依存) | **凍結** | ρ 選択が結果支配 → IG または狭範囲再評価 |
| C2 or C3 失敗 (real 自体が不安定) | **凍結** | candidate 規則 (Phase 2) 見直し |

## 4. Cherry picking 回避 commitments

1. **3 ρ 全結果報告**: 個別に削除しない、Pearson 最良 ρ を後付け選定しない
2. **採否規則の事後変更禁止**: §3.3 / §3.5 を実験後に変更しない (修正は v001.2 として別 pre-registration)
3. **追加 ρ 値の事後追加禁止** (例: 0.1, 0.4, 0.6, 0.9 を結果見て追加しない)
4. **rank-based fallback 採用しない**: v001 で行ったが v001.1 では C2-C5 全通過のみで進入
5. **失敗を成功として再解釈しない** (例: 「Pearson 通らなかったが Spearman 良いから OK」のような事後緩和禁止)

## 5. 報告コミット

実験完了後の public artifact:

- 全 3 ρ の linear attribution parquet (gitignored、hash_log で sha256 audit)
- `data/profiling/v001/phase3_rho_sweep_validation.json`: per-ρ pearson/spearman/rel_err + 同一 500 pair の cross-ρ real TE 配列
- `data/profiling/v001/phase3_rho_sweep_decision.json`: C1-C5 計算 + Phase 4 採否判定
- per-layer Pearson breakdown (gate 化しない、Marks 2024 の早期 layer AtP 弱の文脈確認用)
- 失敗時は failure 構造を明記し v001.2 pre-registration へ link

## 6. Falsifiability

| Falsifier | 検出条件 | 含意 |
|---|---|---|
| **F1** | C5 単独失敗 | 線形近似が本 setup で不適 → IG 必須 |
| **F2** | C4 失敗 | 線形近似精度が摂動強度に強く依存 → 単一 ρ での結果は arbitrary |
| **F3** | C2 or C3 失敗 | real effect 自体が ρ で qualitative に変化 → candidate 規則の semantic に問題 |

## 7. 実装 commitments

1. 本文書を git commit (時系列証跡確定)
2. `experiments/phase3_attribution_pythia_160m.py` を `RHO_SWEEP=[0.25, 0.5, 0.75]` ループ + `N_VALIDATION_PAIRS=500` に改造、commit
3. `experiments/phase3_rho_sweep_decide.py` 新規作成: 3 ρ 結果から C1-C5 計算 + Phase 4 採否判定
4. 1 GPU job で 3 ρ をシーケンシャル実行 (推定 wall ~10 min)
5. decision.json 出力 → 採否判定を本文書 §3.5 に従い決定 → report

**この pre-registration は本 commit 時点で凍結される。以降の変更は v001.2 / v001.3 として別文書を切る。**
