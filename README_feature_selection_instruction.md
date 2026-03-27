# 介入候補特徴量選定スクリプト - 技術仕様書

## 概要

本リポジトリには、SAE特徴量のAtP（Attribution Patching）スコアと活性値データから、LLMの迎合性（Sycophancy）を抑制するための**介入候補特徴量を選定するスクリプト**が含まれています。

**2つの選定手法:**

### 1. 統合的選定: `select_intervention_features.py`

全template_typeを統合してAtPスコアを計算し、全体的に強い影響を持つ特徴量を選定します。

**目的:**
- 迎合時に特異的に働く特徴量を特定
- 言語能力への副作用を最小化
- 因果効果（AtP）が高い順にランキング

**適用場面:**
- 全てのtemplate_typeで共通して機能する特徴量を探す場合
- 全体的な傾向を把握したい場合

### 2. Template Type別選定: `select_intervention_features_per_template.py` ⭐ NEW

各template_type（"I really like", "I think"など）ごとに独立してAtPスコアを計算し、各typeで上位15個を選定後、それらを統合します。

**目的:**
- template_typeごとに異なる迎合特徴を捉える
- 各typeで特異的に働く特徴量を漏らさず選定
- より包括的な介入候補リストを作成

**適用場面:**
- template_typeごとで迎合メカニズムが異なる可能性がある場合
- 介入実験で効果が見られない場合の改善策として
- より網羅的な特徴量選定が必要な場合

**データソース:**
- `atp_results_gemma-2-9b-it_YYYYMMDD_HHMMSS.json`: AtP分析済みデータ
- 指定したトークン位置（デフォルト: `prompt_last_token`）の活性値を使用

**データセットサンプリング機能 🆕:**
- `arguments`データセットから250問（デフォルト）
- `math`データセットから250問（デフォルト）
- 合計500問を用いて分析を実施
- `--sample_per_dataset`オプションでサンプル数をカスタマイズ可能（0で全問使用）

---

## 1. 入力データ構造

### JSONファイルの構造

```json
{
  "metadata": {
    "model_name": "gemma-2-9b-it",
    "sae_release": "gemma-scope-9b-it-res-canonical",
    "num_questions": 100,
    ...
  },
  "results": [
    {
      "question_id": 0,
      "dataset": "arguments",
      "base_text": "質問文の一部...",
      "variations": [
        {
          "template_type": "base",
          "prompt": "質問全文",
          "response": "中立的な回答",
          "sycophancy_flag": 0,
          "sae_activations": {
            "prompt_last_token": {
              "1234": 1.2,
              "5678": 0.5,
              ...
            },
            "response_first_token": { ... }
          }
        },
        {
          "template_type": "I really like",
          "prompt": "質問全文 + I really like...",
          "response": "迎合的な回答",
          "sycophancy_flag": 1,
          "sae_activations": {
            "prompt_last_token": {
              "1234": 2.5,
              "5678": 0.1,
              ...
            }
          },
          "atp_analysis": {
            "target_token": "Yes",
            "base_token": "There",
            "token_position": 1,
            "logit_diff": 3.85,
            "top_features": [
              {
                "id": "1234",
                "score": 4.2,         // AtPスコア
                "activation": 2.5,    // 活性値（参考、sae_activationsと同じ）
                "gradient": 3.5       // 勾配
              },
              ...
            ]
          }
        }
      ]
    }
  ]
}
```

**重要な構造の変更:**
- **全バリエーションに`sae_activations`が存在**: Base時も迎合時も活性値データあり
- **複数のトークン位置**: `prompt_last_token`, `response_first_token`など
- **AtPスコアは迎合時のみ**: `atp_analysis`は迎合時（`sycophancy_flag: 1`）のみに存在

---

## 2. 指標の算出方法

### 2.1 Global Mean AtP（全体平均AtPスコア）

**定義:**
全迎合サンプルに対するAtPスコアの平均値。各特徴量が迎合挙動に与える因果効果を定量化します。

**計算式:**
```
Global Mean AtP = (1/N) × Σ(AtP_i)
```
- `N`: 迎合サンプル数（sycophancy_flag=1のサンプル数）
- `AtP_i`: 各サンプルiにおけるAtPスコア

**AtPスコアの定義:**
```
AtP Score = Activation × Gradient
```
- `Activation`: 特徴量の活性値
- `Gradient`: ターゲット指標（Logit Difference）に対する勾配

**意味:**
- **正の値**: 迎合を促進する特徴（介入候補）
- **負の値**: 迎合を抑制する特徴（保護すべき）
- **絶対値が大きい**: 因果効果が強い

---

### 2.2 Mean Activation (平均活性値)

#### 2.2.1 Mean Activation Syc（迎合時平均活性値）

**定義:**
全迎合サンプル（sycophancy_flag=1）における特徴量の平均活性値。活性化しなかったサンプルは0として扱います。

**計算式:**
```
Mean Activation Syc = (1/N_syc) × Σ(Activation_i)
```
- `N_syc`: 全迎合サンプル数（活性化しなかったサンプルも含む）
- `Activation_i`: 各迎合サンプルでの活性値（活性化しなかった場合は0）

**重要な計算上の注意:**
- SAE特徴量はスパースであるため、多くのサンプルで活性値が0になります
- この平均値は**全サンプル**を母集団とし、活性化しなかったサンプルの寄与を0として含めます
- 例: 100サンプル中20サンプルのみ活性化（平均活性値5.0）の場合
  - Mean Activation Syc = (20 × 5.0 + 80 × 0) / 100 = 1.0

**意味:**
- 迎合的な回答生成時に、その特徴量が**平均的に**どれだけ強く発火しているか
- 活性化頻度とその強度の両方を反映した指標

---

#### 2.2.2 Mean Activation Base（Base時平均活性値）

**定義:**
全Baseサンプル（template_type="base"）における特徴量の平均活性値。活性化しなかったサンプルは0として扱います。

**計算式:**
```
Mean Activation Base = (1/N_base) × Σ(Activation_j)
```
- `N_base`: 全Baseサンプル数（活性化しなかったサンプルも含む）
- `Activation_j`: 各Baseサンプルでの活性値（活性化しなかった場合は0）

**重要な計算上の注意:**
- Mean Activation Sycと同様、全サンプルを母集団とした平均値
- スパース性により、多くのサンプルで活性値が0になることを考慮

**意味:**
- 中立的な回答生成時（通常の言語処理）に、その特徴量が**平均的に**どれだけ使われているか
- 値が大きい → 通常の言語能力に頻繁に必要とされる特徴の可能性（介入すると副作用リスク）
- 値が小さい → 通常時はほとんど使われない特徴（介入の副作用リスク低）

---

### 2.3 Log Ratio（迎合特異性指標）

**定義:**
迎合時とBase時の平均活性値の比の対数。特徴量が迎合に特異的かどうかを示します。

**計算式:**
```
Log Ratio = log₂((Mean Activation Syc + ε) / (Mean Activation Base + ε))
```
- `ε = 1e-6`: ゼロ除算防止用の微小値
- `Mean Activation Syc`: 全迎合サンプルでの平均活性値（活性化しなかった場合は0）
- `Mean Activation Base`: 全Baseサンプルでの平均活性値（活性化しなかった場合は0）
- **log₂を使用**: 機械学習の標準で、fold-change分析と一貫性がある

**なぜ全サンプルベースの平均が重要か:**
真の「迎合特異性」を測定するには、活性化しなかったサンプルも含めた全体像を把握する必要があります。

**具体例による比較:**
- 全迎合サンプル数: 100、全Baseサンプル数: 100
- 特徴Aの迎合時活性化: 20サンプル（活性値平均5.0）、Base時活性化: 5サンプル（活性値平均2.0）

**正しい計算（全サンプルベース）:**
- Mean Activation Syc = (20 × 5.0) / 100 = 1.0
- Mean Activation Base = (5 × 2.0) / 100 = 0.1
- Log Ratio = log₂(1.0 / 0.1) = 3.32 → 真に迎合特異的（10倍の変化）

**誤った計算（活性化したサンプルのみ）:**
- 迎合時平均 = 5.0、Base時平均 = 2.0
- Log Ratio = log₂(5.0 / 2.0) = 1.32 → 特異性を過小評価

**解釈:**
| Log Ratio (log₂) | 活性値比 | 意味 |
|-----------------|----------|------|
| < 0 | < 1倍 | Base時により強く発火（迎合時は抑制）|
| 0.0 | 1倍 | 迎合時もBase時も同程度 |
| 0.5 | 1.41倍 | 迎合時にやや強く発火 |
| 1.0 | 2倍 | 迎合時に2倍強く発火（推奨閾値） |
| 1.5 | 2.83倍 | 迎合時に約3倍強く発火 |
| 2.0 | 4倍 | 迎合時に4倍強く発火 |
| 3.0 | 8倍 | 迎合時に8倍強く発火 |

**重要性:**
- **Log Ratio が高い** → 迎合特異的 → 介入しても言語能力への副作用が少ない
- **Log Ratio が低い** → 通常時も使用 → 介入すると言語能力が崩壊する恐れ
- **Log Ratio が負** → 通常時により使用 → 介入すると言語能力が大幅に低下

---

## 3. フィルタリングロジック

### 3.1 複合フィルタ（AND条件）

選定される特徴量は、以下の**3つすべて**を満たす必要があります：

#### 条件1: Positive Impact（正の因果効果）
```
global_mean_atp > 0
```
- **理由**: 負の値は迎合を抑制している良い特徴なので、除外して保護する

#### 条件2: High Specificity（高い迎合特異性）
```
log_ratio > 1.0  （デフォルト値）
```
- **理由**: 通常の言語処理で使われる特徴を消すと、言語能力が崩壊する
- **推奨値**: 1.0（2倍以上の特異性）

#### 条件3: Minimum Impact（最小影響力）
```
global_mean_atp > 1e-4  （デフォルト値）
```
- **理由**: ノイズレベルの微小な寄与を除外

---

### 3.2 ランキングとTop-K選定

フィルタを通過した特徴量を、`global_mean_atp` の**降順（大きい順）**にソートし、上位K個を選定します。

**デフォルト値**: `K = 50`

---

## 4. 出力データ構造

### 4.1 CSVファイル: `intervention_candidates_YYYYMMDD_HHMMSS.csv`

選定された特徴量の詳細データ。

**カラム構成:**

| カラム名 | データ型 | 説明 |
|---------|---------|------|
| `feature_index` | int | 特徴量ID |
| `global_mean_atp` | float | 全体平均AtPスコア（因果効果） |
| `conditional_mean_atp` | float | 活性化した時のみの平均AtPスコア（参考値） |
| `mean_activation_syc` | float | 迎合時平均活性値（全サンプルベース） |
| `mean_activation_base` | float | Base時平均活性値（全サンプルベース） |
| `log_ratio` | float | 迎合特異性指標 |
| `num_samples_active_syc` | int | 迎合時に活性化したサンプル数 |
| `num_samples_active_base` | int | Base時に活性化したサンプル数 |
| `num_samples_total_syc` | int | 総迎合サンプル数 |
| `num_samples_total_base` | int | 総Baseサンプル数 |
| `activation_rate_syc` | float | 迎合時活性化率 |
| `activation_rate_base` | float | Base時活性化率 |

**ソート順**: `global_mean_atp` 降順

**サンプル:**
```csv
feature_index,global_mean_atp,conditional_mean_atp,mean_activation_syc,mean_activation_base,log_ratio,num_samples_active_syc,num_samples_active_base,num_samples_total_syc,num_samples_total_base,activation_rate_syc,activation_rate_base
1234,5.234,6.012,1.456,0.123,1.073,87,12,100,100,0.87,0.12
5678,4.891,5.321,2.012,0.089,1.354,92,8,100,100,0.92,0.08
9012,3.567,4.693,0.890,0.134,0.822,76,15,100,100,0.76,0.15
...
```

---

### 4.2 テキストファイル: `selection_summary_YYYYMMDD_HHMMSS.txt`

選定結果の統計サマリー。全特徴量の統計と選定された特徴量の詳細を含む。

**内容:**
```
=== 介入候補特徴量 選定サマリー ===

--- 実行パラメータ ---
入力ファイル: feedback_results/combined_feedback_data.json
トークン位置: prompt_last_token
選定数: Top-50
最小AtPスコア: 0.0001
最小Log Ratio: 1.0

============================================================
--- 全特徴量の統計情報 ---
総特徴量数: 16384

主要指標の統計量:
       global_mean_atp  log_ratio  mean_activation_syc  mean_activation_base  ...
count        16384.000  16384.000            16384.000             16384.000  ...
mean             0.045      0.321                0.123                 0.098  ...
std              0.234      1.456                0.456                 0.234  ...
min             -2.345     -3.210                0.000                 0.000  ...
25%             -0.012     -0.543                0.012                 0.008  ...
50%              0.034      0.234                0.045                 0.034  ...
75%              0.123      1.123                0.134                 0.123  ...
max              5.678      4.567                3.456                 2.345  ...

Global Mean AtPの分布:
  正の値: 8234 特徴 (50.3%)
  負の値: 7890 特徴 (48.2%)
  ゼロ: 260 特徴

Log Ratioの分布:
  > 2.0 (4倍): 1234 特徴
  > 1.0 (2倍): 3456 特徴
  > 0.5 (1.4倍): 5678 特徴
  > 0.0: 8901 特徴
  < 0.0 (Base時により活性化): 7483 特徴

============================================================

--- 選定された特徴量の統計情報 ---
選定数: 50 特徴量

       global_mean_atp  log_ratio  mean_activation_syc  mean_activation_base
count        50.000000  50.000000            50.000000             50.000000
mean          2.345678   1.876543             1.234567              0.123456
std           1.234567   0.634567             0.567890              0.067890
min           1.000000   1.000000             0.500000              0.010000
...

--- 上位10特徴量 ---
   feature_index  global_mean_atp  log_ratio  ...
0           1234         5.234000   3.301000  ...
1           5678         4.891000   2.529000  ...
...
```

**含まれる情報:**

1. **実行パラメータ**: 入力ファイル、トークン位置、フィルタ閾値
2. **全特徴量の統計**:
   - 総特徴量数
   - 主要指標（AtP、Log Ratio、活性値など）の記述統計
   - Global Mean AtPの分布（正・負・ゼロの割合）
   - Log Ratioの分布（特異性レベル別の件数）
3. **選定特徴の統計**: フィルタ通過後の統計量
4. **上位特徴の詳細**: 上位10件の詳細データ

---

### 4.3 画像ファイル: `intervention_selection_YYYYMMDD_HHMMSS.png`

散布図による可視化。

**軸:**
- **横軸（X）**: `log_ratio` (迎合特異性)
- **縦軸（Y）**: `global_mean_atp` (因果効果)

**プロット:**
- **グレー点**: 全特徴量
- **赤色点**: 選定された特徴量（赤枠付き）
- **破線（青）**: Log Ratio閾値（デフォルト: 0.5）
- **破線（黒）**: AtP = 0 のライン

**解釈:**
- **右上の領域**: 高特異性 & 高因果効果 → 理想的な介入候補
- **左上の領域**: 低特異性 & 高因果効果 → 副作用リスクあり（除外される）
- **右下の領域**: 高特異性 & 低因果効果 → 効果が弱い
- **左下の領域**: 低特異性 & 低因果効果 → 介入不要

---

### 4.4 Template Type別選定の出力ファイル ⭐ NEW

`results/selection_results_per_template/`ディレクトリに以下が保存されます：

#### 1. `merged_intervention_candidates_{timestamp}.csv`

**統合された特徴量IDリスト**（介入実験で使用）

```csv
feature_index
1234
5678
9012
...
```

- 全template_typeの選定結果を統合
- 重複を除いたユニークなリスト
- **このファイルを介入実験に使用します**

#### 2. `candidates_{template_type}_{timestamp}.csv`

各template_typeごとの詳細データ（複数ファイル生成）

```csv
feature_index,template_type,global_mean_atp,conditional_mean_atp,mean_activation_syc,mean_activation_base,log_ratio,...
1234,I really like,5.234,6.012,1.456,0.123,3.073,...
5678,I really like,4.891,5.321,2.012,0.089,2.354,...
...
```

- template_typeごとに選定された特徴量の詳細
- 各template_typeでの統計情報を含む

#### 3. `selection_summary_{timestamp}.txt`

選定サマリーと重複分析

```
============================================================
Template Type別 介入候補特徴量 選定サマリー
============================================================

--- 実行パラメータ ---
入力ファイル: atp_calculated_results/atp_results_gemma-2-9b-it_20251201_095948.json
トークン位置: prompt_last_token
各template_typeでの選定数: 15
最小AtPスコア: 0.0
最小Log Ratio: 0.0
各datasetからのサンプル数: 250 🆕

--- 統合結果 ---
template_type数: 4
合計選定数（重複あり）: 60
ユニークな特徴量数: 45

--- template_typeごとの選定状況 ---

I really like:
  選定数: 15
  AtP範囲: 0.005234 ~ 0.001234
  Log Ratio範囲: 3.07 ~ 1.23
  上位5特徴量:
    - Feature 1234: AtP=0.005234, LogRatio=3.07
    - Feature 5678: AtP=0.004891, LogRatio=2.35
    ...

I think:
  選定数: 15
  ...

--- 統合された特徴量IDリスト（全て） ---
[1234, 5678, 9012, ...]

--- 重複分析 ---
重複していない特徴（1つのtemplateのみ）: 30
2つのtemplateで選ばれた特徴: 10
3つのtemplateで選ばれた特徴: 3
4つのtemplateで選ばれた特徴: 2

全template_typeで選ばれた共通特徴（2個）:
[1234, 5678]
```

#### 4. `per_template_selection_{timestamp}.png`

template_typeごとの散布図（2×2グリッド）

- 各template_typeでの選定結果を個別に可視化
- 横軸: Log Ratio、縦軸: Global Mean AtP
- 各typeで異なる選定パターンを確認可能

#### 5. `overlap_analysis_{timestamp}.png`

重複状況の棒グラフ

- 横軸: 重複度（1つのtemplate、2つのtemplate、...）
- 縦軸: 特徴量数
- どれだけの特徴が複数のtemplate_typeで選ばれたかを可視化

---

## 5. 使用例

### 5.1 統合的選定（従来の方法）

#### 基本実行
```bash
# デフォルト設定（prompt_last_token使用）
python select_intervention_features.py

# カスタム設定
python select_intervention_features.py \
  --input feedback_results/combined_feedback_data.json \
  --token_position prompt_last_token \
  --top_k 50
```

#### トークン位置の指定
```bash
# プロンプト最終トークン（デフォルト）
python select_intervention_features.py \
  --token_position prompt_last_token

# レスポンス最初のトークン
python select_intervention_features.py \
  --token_position response_first_token
```

### 5.2 Template Type別選定（推奨）⭐ NEW

#### 基本実行
```bash
# デフォルト設定（各template_typeでトップ15、合計最大60個）
# デフォルトで各datasetから250問ずつサンプリング（合計500問）
python select_intervention_features_per_template.py

# 各template_typeでの選定数を変更
python select_intervention_features_per_template.py --top_k_per_template 20

# 入力ファイルを指定
python select_intervention_features_per_template.py \
  --input atp_calculated_results/atp_results_gemma-2-9b-it_20251201_095948.json
```

#### データセットサンプリングのカスタマイズ 🆕
```bash
# デフォルト（各datasetから250問ずつ、合計500問）
python select_intervention_features_per_template.py

# 各datasetから100問ずつサンプリング（合計200問）
python select_intervention_features_per_template.py --sample_per_dataset 100

# 各datasetから500問ずつサンプリング（合計1000問）
python select_intervention_features_per_template.py --sample_per_dataset 500

# 全問使用（サンプリングなし）
python select_intervention_features_per_template.py --sample_per_dataset 0
```

#### フィルタリング条件の調整
```bash
# より厳格な選定（高特異性、高影響力）
python select_intervention_features_per_template.py \
  --top_k_per_template 15 \
  --min_atp 1e-5 \
  --min_log_ratio 0.5

# より緩い選定（候補を多く）
python select_intervention_features_per_template.py \
  --top_k_per_template 20 \
  --min_atp 0.0 \
  --min_log_ratio 0.0
```

#### 出力ディレクトリの指定
```bash
python select_intervention_features_per_template.py \
  --output_dir results/selection_results_per_template/experiment_001
```

#### パラメータ調整（統合的選定）
```bash
# より厳格な選定（高特異性、高影響力）
python select_intervention_features.py \
  --input feedback_results/combined_feedback_data.json \
  --token_position prompt_last_token \
  --top_k 30 \
  --min_atp 5e-4 \
  --min_log_ratio 2.0

# より緩い選定（候補を多く）
python select_intervention_features.py \
  --input feedback_results/combined_feedback_data.json \
  --token_position prompt_last_token \
  --top_k 100 \
  --min_atp 1e-5 \
  --min_log_ratio 0.5
```

### 5.3 コマンドライン引数

#### 統合的選定（select_intervention_features.py）

| 引数 | デフォルト値 | 説明 |
|------|-------------|------|
| `--input` | feedback_results/combined_feedback_data.json | combined_feedback_data.jsonのパス |
| `--token_position` | prompt_last_token | 使用するトークン位置（prompt_last_token, response_first_tokenなど） |
| `--top_k` | 50 | 選定する上位特徴量数 |
| `--min_atp` | 1e-4 | 最小AtPスコア閾値 |
| `--min_log_ratio` | 1.0 | 最小Log Ratio閾値（1.0 = 2倍の特異性） |
| `--output_dir` | results/selection_results | 結果の保存先ディレクトリ |

#### Template Type別選定（select_intervention_features_per_template.py）⭐ NEW

| 引数 | デフォルト値 | 説明 |
|------|-------------|------|
| `--input` | atp_calculated_results/atp_results_gemma-2-9b-it_20251201_095948.json | atp_results.jsonのパス |
| `--token_position` | prompt_last_token | 使用するトークン位置 |
| `--top_k_per_template` | 15 | 各template_typeで選定する特徴量数 |
| `--min_atp` | 0.0 | 最小AtPスコア閾値（0.0 = 正の値のみ） |
| `--min_log_ratio` | 0.0 | 最小Log Ratio閾値 |
| `--sample_per_dataset` 🆕 | 250 | 各dataset(arguments/math)から取得する問題数（0で全問使用） |
| `--output_dir` | results/selection_results_per_template | 結果の保存先ディレクトリ |

---

## 6. 処理フロー

### 6.1 統合的選定（select_intervention_features.py）

```
[Step 1] データ読み込み
    ↓
    - combined_feedback_data.jsonからデータ読み込み
    - 指定されたtoken_positionの活性値を抽出
    - Base時: sae_activationsから活性値取得
    - 迎合時: sae_activationsから活性値 + atp_analysisからAtPスコア取得
    - 特徴量ごとに集約
    
[Step 2] Log Ratio計算
    ↓
    - 各特徴量の迎合特異性を計算
    
[Step 3] 複合フィルタリング
    ↓
    - 条件1: global_mean_atp > 0
    - 条件2: log_ratio > min_log_ratio
    - 条件3: global_mean_atp > min_atp
    
[Step 4] Top-K選定
    ↓
    - AtPスコア降順でソート
    - 上位K個を選定
    
[Step 5] 結果保存
    ↓
    - CSV: 選定特徴量データ
    - TXT: 統計サマリー
    
[Step 6] 可視化
    ↓
    - PNG: 散布図（Log Ratio vs AtP）
```

### 6.2 Template Type別選定（select_intervention_features_per_template.py）⭐ NEW

```
[Step 1] データ読み込みとtemplate_type別集計
    ↓
    - atp_results.jsonからデータ読み込み
    - **データセットサンプリング（デフォルト: 各250問）🆕**
      ・argumentsデータセットから250問をサンプリング
      ・mathデータセットから250問をサンプリング
      ・合計500問を使用して分析
    - 迎合誘発template_typeを検出（base以外）
    - 各template_typeごとに独立して以下を実行:
      ・そのtypeの迎合サンプル数とbaseサンプル数をカウント
      ・そのtypeでのAtPスコアと活性値を集計
      ・Global Mean AtPとLog Ratioを計算
    
[Step 2] 各template_typeでトップK選定
    ↓
    - 各template_typeごとに:
      ・条件1: global_mean_atp > min_atp（デフォルト: 0）
      ・条件2: log_ratio > min_log_ratio（デフォルト: 0）
      ・AtPスコア降順でソート
      ・上位K個（デフォルト: 15）を選定
    
[Step 3] リストの統合
    ↓
    - 全template_typeの選定結果を統合
    - 重複を除いてユニークなリストを作成
    - 重複状況を分析（何個のtemplateで選ばれたか）
    
[Step 4] 結果保存
    ↓
    - CSV: 統合リスト（merged_intervention_candidates）
    - CSV: 各template_typeの詳細データ
    - TXT: 選定サマリーと重複分析
    
[Step 5] 可視化
    ↓
    - PNG: template_type別散布図（2×2グリッド）
    - PNG: 重複状況の棒グラフ
```

---

## 7. 理論的背景

### 7.1 なぜLog Ratioが重要か？

**問題:**
単にAtPスコアが高い特徴量を消すと、言語能力も損なわれる可能性がある。

**解決策:**
「迎合時」には強く働くが、「平時（Base）」にはあまり働かない**特異的な特徴量**のみを選定する。

**具体例（全サンプルベースで正しく計算）:**

全サンプル数: 迎合100、Base100

- **特徴A**: AtP=5.0
  - 迎合時: 20サンプルで活性化（平均5.0） → Mean Activation Syc = 1.0
  - Base時: 2サンプルで活性化（平均5.0） → Mean Activation Base = 0.1
  - Log Ratio = log₂(1.0 / 0.1) = 3.32 ✓ **迎合特異的（10倍） → 介入候補**

- **特徴B**: AtP=5.0
  - 迎合時: 80サンプルで活性化（平均3.1） → Mean Activation Syc = 2.5
  - Base時: 80サンプルで活性化（平均3.0） → Mean Activation Base = 2.4
  - Log Ratio = log₂(2.5 / 2.4) = 0.06 ✗ **通常時も使用 → 介入すると言語崩壊**

特徴AとBは同じAtPスコアですが：
- **特徴A**: 迎合時のみ選択的に活性化（活性化率: 20% vs 2%）→ 安全に介入可能
- **特徴B**: 常時活性化（活性化率: 80% vs 80%）→ 介入すると言語能力が崩壊

このように、全サンプルベースでの平均活性値を用いることで、真の「特異性」を正しく評価できます。

---

### 7.2 AtPスコアの因果的解釈

AtP（Attribution Patching）は、モデルの勾配情報を用いて因果効果を一次近似します。

```
Score = Activation × Gradient
```

**直感的理解:**
- **Activation**: その特徴がどれだけ強く働いたか
- **Gradient**: その特徴が1単位増えたとき、ターゲット（Logit Difference）がどれだけ変化するか
- **Score**: 両者の積 = その特徴がターゲットに与えた因果的寄与

**メトリック（ターゲット）:**
```
Logit Difference = Logit(Target Token) - Logit(Base Token)
```
- 迎合的な回答トークンと中立的な回答トークンのロジット差
- この値が大きいほど、モデルは迎合的な回答を選びやすくなる

---

## 8. データ要件

### 必須データ

1. **combined_feedback_data.json**:
   - 全バリエーション（Base時・迎合時）に`sae_activations`が必要
   - 迎合時には追加で`atp_analysis`が必要

2. **トークン位置**:
   - `sae_activations`内に指定したトークン位置のデータが必要
   - 利用可能なトークン位置: `prompt_last_token`, `response_first_token`など

### データの一貫性

- Base時と迎合時で**同じトークン位置**の活性値を使用
- 特徴量IDは文字列または整数で一貫している必要がある

### 旧データとの互換性

**注意**: このスクリプトは`atp_results.json`（Base時に`sae_activations`がないデータ）とは**互換性がありません**。`combined_feedback_data.json`形式のデータを使用してください。

---

## 9. 注意事項とベストプラクティス

### 8.1 パラメータ調整のガイドライン

#### 統合的選定の場合

**`min_log_ratio` の設定:**
- **0.5**: 緩い（約1.4倍の特異性）→ 候補が多い、副作用リスク中
- **1.0**: 推奨（2倍の特異性）→ バランス良好
- **1.5**: やや厳格（約3倍の特異性）→ 安全性重視
- **2.0**: 厳格（4倍の特異性）→ 候補が少ない、副作用リスク低

**`top_k` の設定:**
- **小（20-30）**: 最も効果的な特徴のみ → 保守的介入
- **中（50-100）**: バランス → 標準的介入
- **大（100-200）**: 広範囲介入 → 効果は大きいが副作用リスク増

#### Template Type別選定の場合 ⭐ NEW

**`top_k_per_template` の設定:**
- **小（10-15）**: 各typeで最も効果的な特徴のみ → 保守的（合計40-60個程度）
- **中（15-20）**: バランス良好 → 標準的（合計60-80個程度）
- **大（20-30）**: 各typeで広範囲選定 → 包括的（合計80-120個程度）

**注意**: 最終的な統合リストの特徴量数は、重複度により変動します。重複が多い場合は統合後の数が少なくなります。

**`min_log_ratio` と `min_atp` の設定:**
- **デフォルト（0.0）**: 正の値のみを選定 → 各typeの特徴を広く捉える
- **厳格（0.5, 1e-5）**: 高特異性・高影響力のみ → 安全性重視

---

### 8.2 手法の選択ガイド

#### 統合的選定を使うべき場合:
- 初回の探索的分析
- 全template_typeで共通して機能する汎用的な特徴を探す場合
- シンプルな解釈を優先する場合

#### Template Type別選定を使うべき場合（推奨）⭐:
- **介入実験で効果が見られない場合**
- template_typeごとで異なる迎合メカニズムが疑われる場合
- より包括的で確実な特徴量選定が必要な場合
- 各template_typeの特性を詳しく分析したい場合

---

### 8.3 結果の検証方法

#### 統合的選定の場合:
1. **散布図の確認**: 選定された特徴が右上に集中しているか
2. **Log Ratioの分布**: 十分に高い特異性（>1.0）を持つか
3. **Base活性値の確認**: 通常時の活性値が低いか（副作用リスク評価）

#### Template Type別選定の場合 ⭐ NEW:
1. **template_type別散布図の確認**: 各typeで選定パターンが異なるか
2. **重複分析の確認**: 
   - 全typeで選ばれた共通特徴 → 汎用的な迎合特徴
   - 1つのtypeのみで選ばれた特徴 → type特異的な迎合特徴
3. **統合リストのサイズ確認**: 期待通りの特徴量数が得られているか
4. **各typeの統計情報**: AtP範囲とLog Ratio範囲が妥当か

---

## 9. 次のステップ（Ablation実験）

選定された特徴量を用いて、実際に介入実験を行います。

**推奨ワークフロー**:

1. **初回実験**: 統合的選定で全体傾向を把握
   ```bash
   python select_intervention_features.py --top_k 50
   # → 介入実験
   ```

2. **改善実験**: Template Type別選定で包括的に対応 ⭐
   ```bash
   # デフォルト設定（各datasetから250問ずつ、合計500問を使用）
   python select_intervention_features_per_template.py --top_k_per_template 15
   
   # サンプル数をカスタマイズ 🆕
   python select_intervention_features_per_template.py \
     --top_k_per_template 15 \
     --sample_per_dataset 300  # 各datasetから300問ずつ（合計600問）
   
   # → merged_intervention_candidates_*.csv を使用して介入実験
   ```

**手法**: Geometric Subtraction (Zero-Ablation)
```python
x' = x - (Activation(f_i) × d_i)
```
- 特定特徴量の方向ベクトルを残差ストリームから削除
- **Template Type別選定の統合リストをそのまま使用**（全template_typeで同じ特徴を介入）

**評価指標:**
1. **Sycophancy Rate**: 迎合挙動の減少（McNemar検定で有意性を確認）
2. **Naturalness Score**: 言語能力の保持（副作用評価）

---

## 10. ライセンスと引用

このスクリプトは研究目的で作成されています。使用する際は、SAE-LensおよびTransformerLensの引用を含めてください。

**関連論文:**
- Templeton et al. (2024) "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet"
- Marks et al. (2024) "Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models"
