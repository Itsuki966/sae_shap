# SAE介入特徴探索プログラム - 完全ガイド

## 📋 概要

このプログラムは、LLMの迎合性（Sycophancy）抑制のために介入すべきSAE特徴を特定します。
機械学習モデル（LightGBM）、SHAP値分析、多層的検証を組み合わせて、**安全で効果的な介入ターゲット**を発見します。

### 主な特徴

- ✅ **5段階フィルタリング**で信頼性の高い特徴を特定
- ✅ **活性化頻度分析**で実用的な特徴を選択
- ✅ **予測確率レベル別分析**で介入の優先度を判断
- ✅ **誤分類分析（FP/FN/TP/TN）**でノイズ特徴を除外
- ✅ **完全なトレーサビリティ**でフィルタリング過程を記録

---

## 🚀 クイックスタート

### 基本的な使い方

```bash
python find_intervention_features.py \
  --input combined_feedback_data.json \
  --token_position prompt_last_token
```

### 引数

| 引数 | 短縮形 | 必須 | デフォルト | 説明 |
|------|--------|------|-----------|------|
| `--input` | `-i` | ❌ | `combined_feedback_data.json` | 入力JSONファイル |
| `--token_position` | `-t` | ✅ | - | 分析対象のトークン位置 |
| `--output` | `-o` | ❌ | `results` | 結果の保存先 |

### 使用例

```bash
# 例1: プロンプト最終トークンを分析
python find_intervention_features.py -t prompt_last_token

# 例2: レスポンス最初のトークンを分析
python find_intervention_features.py -t response_first_token

# 例3: カスタム出力ディレクトリを指定
python find_intervention_features.py \
  -i data/experiment_v2.json \
  -t prompt_last_token \
  -o results_v2
```

---

## 📊 分析フロー

### 1. データ読み込みと前処理

```
入力JSONファイル
  ↓
SAE特徴抽出 (sae_activations)
  ↓
フラグ -1 のサンプルを除外
  ↓
DataFrame化 (疎行列処理)
```

**出力**: 特徴行列 X (n_samples × n_features)

---

### 2. モデル学習とSHAP値計算

```
5-Fold Stratified Cross-Validation
  ↓
各Foldで LightGBM 学習
  ↓
SHAP TreeExplainer で寄与度計算
  ↓
全Foldの結果を統合・整列
```

**重要**: 各Foldの性能指標（Accuracy, F1, ROC AUC等）を記録

---

### 3. 多層的検証による介入特徴の特定

#### 🔍 **5段階フィルタリングプロセス**

| ステップ | 基準 | 目的 |
|---------|------|------|
| **1. 量的基準** | 重要度 > 90パーセンタイル | 重要な特徴を選別 |
| **2. 方向性** | 平均SHAP > 0 | 迎合性を促進する特徴のみ |
| **3. 一貫性** | 正の寄与率 > 70% | 予測可能な効果 |
| **4. TP検証** | TP平均SHAP > 0.01 | 実際の迎合で有効 |
| **5. 重複除去** | 相関 < 0.9 | 冗長性を排除 |

**トレーサビリティ**: 各ステップの候補数と除外理由を記録

---

### 4. 補完的分析

#### 📈 **活性化頻度分析**

```python
# 各特徴について:
activation_frequency = (特徴値 > 0.01).sum() / 総サンプル数
mean_value_when_active = 活性化時の平均値
```

**目的**: 「ほとんど活性化しない特徴」を検出

#### 🎯 **予測確率レベル別分析**

| レベル | 閾値 | 意味 |
|--------|------|------|
| 高確信度迎合 | ≥ 0.7 | 確実な迎合サンプル |
| 中程度迎合 | 0.3 - 0.7 | 曖昧なケース |
| 非迎合 | < 0.3 | 明確に非迎合 |

各レベルで主要特徴を分析 → **介入の優先度付け**

#### ⚠️ **誤分類分析（FP/FN/TP/TN）**

- **TP (True Positive)**: 確実な介入候補
- **FP (False Positive)**: ノイズ特徴（除外対象）
- **FN (False Negative)**: 見逃された特徴
- **TN (True Negative)**: 非迎合を示す特徴

---

## 📁 出力ファイル構造

```
results/
└── experiments/
    └── {token_position}_{timestamp}/
        ├── config.json                    # 実験設定
        ├── summary.txt                    # テキストサマリー
        ├── data/
        │   ├── cv_results.json            # CVの詳細結果
        │   ├── shap_values.npz            # SHAP値データ
        │   ├── shap_statistics.csv        # ✨ 活性化頻度・効果量含む
        │   ├── top50_features.csv         # 重要特徴ランキング
        │   ├── intervention_features.json # ✨ フィルタリング過程含む
        │   ├── feature_consistency_stats.csv
        │   ├── template_analysis.csv
        │   ├── prediction_level_analysis.csv  # ✨ NEW
        │   └── misclassification_analysis.csv # ✨ NEW
        └── figures/
            ├── 01_model_performance.png
            ├── 02_shap_beeswarm.png
            ├── 03_shap_bar.png
            ├── 04_consistency_analysis.png
            └── 05_template_heatmap.png
```

### ✨ 新規追加ファイル

#### 1. `shap_statistics.csv` に追加された列

| 列名 | 説明 | 用途 |
|------|------|------|
| `activation_frequency` | 活性化頻度（> 0.01） | 実用性の評価 |
| `mean_value_when_active` | 活性化時の平均値 | 活性化強度 |
| `effect_size` | 効果量（標準化SHAP） | 統計的有意性 |

#### 2. `prediction_level_analysis.csv`

```csv
prediction_level,sample_count,rank,feature_name,feature_id,mean_shap,mean_abs_shap
High Confidence Sycophancy (≥0.7),150,1,feature_1234,1234,0.0456,0.0456
High Confidence Sycophancy (≥0.7),150,2,feature_5678,5678,0.0389,0.0389
...
```

**用途**: 確信度別の主要特徴を特定 → 介入の優先順位

#### 3. `misclassification_analysis.csv`

```csv
category,description,rank,feature_name,feature_id,mean_shap,sample_count
TP,確実な介入候補,1,feature_1234,1234,0.0456,120
FP,ノイズ特徴の可能性,1,feature_9999,9999,0.0123,25
...
```

**用途**: ノイズ特徴の検出、介入候補の信頼性評価

#### 4. `intervention_features.json` に追加

```json
{
  "filtering_pipeline": [
    {
      "step": 1,
      "criterion": "importance > 90th percentile (0.012345)",
      "candidates_before": 16384,
      "candidates_after": 1638,
      "removed_count": 14746
    },
    ...
  ]
}
```

**用途**: 論文での説明、パラメータ調整の追跡

---

## 🔬 分析の詳細

### LightGBMハイパーパラメータ

| パラメータ | 値 | 説明 |
|-----------|---|------|
| `objective` | binary | 二値分類 |
| `num_leaves` | 31 | ツリーの複雑さ |
| `learning_rate` | 0.05 | 学習率 |
| `feature_fraction` | 0.9 | 特徴サンプリング |
| `bagging_fraction` | 0.8 | データサンプリング |
| `min_data_in_leaf` | 20 | 過学習防止 |
| `max_depth` | -1 | 深さ制限なし |

### SHAP値の解釈

| SHAP値 | 意味 |
|--------|------|
| > 0 | その特徴が迎合性を**促進** |
| < 0 | その特徴が迎合性を**抑制** |
| ≈ 0 | 影響が小さい/不一致 |

**重要**: 平均SHAP値だけでなく、一貫性（正の寄与率）も確認が必須

---

## 📈 可視化の読み方

### 1. ROC/PR曲線（`01_model_performance.png`）

- **ROC AUC**: モデルの識別能力（高いほど良い）
- **Average Precision**: 不均衡データでの性能（重要）

### 2. SHAP Beeswarm Plot（`02_shap_beeswarm.png`）

⚠️ **注意**: これ**だけ**では介入特徴を決定しない！

| 見た目 | 実際の可能性 | 対策 |
|--------|------------|------|
| 赤い点が右側 | 5%しか活性化しない | 活性化頻度を確認 |
| 明確な色分離 | 50%は負の寄与 | 一貫性スコアを確認 |
| 上位に表示 | 特定テンプレートのみ | テンプレート別分析 |

### 3. 一貫性分析（`04_consistency_analysis.png`）

- **右上象限**: 一貫して迎合性を促進 → **最優先介入候補**
- **右下象限**: 一貫して抑制 → 介入すると逆効果
- **左側**: 文脈依存 → 介入リスク高い

### 4. テンプレート別ヒートマップ（`05_template_heatmap.png`）

- **全テンプレートで赤**: 汎用的な迎合性特徴
- **特定テンプレートのみ赤**: テンプレート特異的

---

## 🎯 介入特徴の選択基準

### ✅ 優れた介入特徴の条件

1. ✔️ **高重要度**: 上位10%の影響力
2. ✔️ **正の寄与**: 平均SHAP > 0
3. ✔️ **高一貫性**: 70%以上のサンプルで正の寄与
4. ✔️ **TP検証済み**: 実際の迎合サンプルで有効
5. ✔️ **適度な活性化**: 5%以上のサンプルで活性化
6. ✔️ **低冗長性**: 他の候補と相関 < 0.9

### ❌ 除外すべき特徴

- ❌ FPで強く寄与（ノイズの可能性）
- ❌ 活性化頻度 < 5%（ほとんど無意味）
- ❌ 一貫性 < 60%（予測不可能）
- ❌ 効果量が小さい（統計的に不安定）

---

## 💡 論文執筆での活用

### Methods セクション

**使用するファイル**:
- `config.json`: モデル設定
- `intervention_features.json` の `filtering_pipeline`: フィルタリング手順

**記載例**:
```
We employed a 5-step filtering process to identify intervention targets:
(1) Feature importance (top 10%), (2) Positive contribution (mean SHAP > 0),
(3) Consistency (>70% positive ratio), (4) True positive validation,
(5) Redundancy removal (correlation < 0.9).
```

### Results セクション

**使用するファイル**:
- `summary.txt`: モデル性能（平均±標準偏差）
- `cv_results.json`: 詳細な統計
- `figures/*.png`: 可視化

**数値の報告**:
```
Cross-validation results (5-fold):
ROC AUC: 0.85 ± 0.03
F1 Score: 0.78 ± 0.04
```

### Tables

| テーブル | ファイル | 内容 |
|---------|---------|------|
| Table 1 | `top50_features.csv` | 重要特徴ランキング |
| Table 2 | `intervention_features.json` | 最終介入特徴リスト |
| Supp. Table 1 | `shap_statistics.csv` | 全特徴の統計 |
| Supp. Table 2 | `prediction_level_analysis.csv` | レベル別分析 |

### Supplementary Materials

- `misclassification_analysis.csv`: 誤分類の詳細分析
- `filtering_pipeline`: フィルタリング過程の完全な記録

---

## 🔧 トラブルシューティング

### エラー1: `maximum feature index in dataset is -1`

**原因**: 指定したトークン位置にSAE特徴が存在しない

**解決策**:
```python
import json
with open("combined_feedback_data.json") as f:
    data = json.load(f)
    first_sample = data["results"][0]["variations"][0]
    print("利用可能なトークン位置:")
    print(list(first_sample["sae_activations"].keys()))
```

### エラー2: メモリ不足

**原因**: 大規模なデータセット（> 10,000サンプル、> 100,000特徴）

**解決策**:
1. サンプル数を削減（代表的なサンプルのみ）
2. 低活性化特徴を事前除外（活性化頻度 < 1%）

### 警告: 介入特徴が少なすぎる（< 5個）

**原因**: フィルタリング基準が厳しすぎる

**調整方法** (`find_intervention_features()` 内):
```python
# 閾値を緩和
importance_threshold = np.percentile(mean_abs_shap, 85)  # 90 → 85
consistency_threshold = 0.65  # 0.7 → 0.65
tp_threshold = 0.005  # 0.01 → 0.005
```

---

## 📚 依存関係

```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
shap>=0.41.0
lightgbm>=3.3.0
scikit-learn>=1.0.0
scipy>=1.7.0
```


## 📖 参考文献

- SHAP: Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"
- LightGBM: Ke et al. (2017) "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
- SAE: Bricken et al. (2023) "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"

---

## 📧 サポート

問題が発生した場合:
1. `config.json` と `summary.txt` を確認
2. `filtering_pipeline` で各ステップの候補数を確認
3. エラーメッセージとデータ統計を記録

---

**最終更新**: 2025-11-25  
**バージョン**: 2.0（優先度高機能実装版）