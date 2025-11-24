# SAE介入特徴探索プログラム

## 概要

このプログラムは、LLMの迎合性（Sycophancy）抑制のために介入すべきSAE特徴を特定するためのツールです。
SHAP値分析、機械学習モデル、複数の可視化手法を組み合わせて、効果的で安全な介入ターゲットを発見します。

## 主な機能

### 1. データ分析パイプライン
- JSONファイルからSAE特徴とsycophancy_flagを抽出
- LightGBMによる二値分類モデルの学習（5-fold cross-validation）
- SHAP値の計算と保存

### 2. 可視化機能

#### ROC/PR曲線分析
- ROC曲線とAUCスコア
- Precision-Recall曲線とAverage Precision
- 最適閾値の探索（F1スコア最大化、Youden's Index）

#### 特徴の一貫性分析
- 一貫性（Consistency）vs 純寄与（Net Contribution）のプロット
- 介入候補特徴のハイライト表示
- 特徴統計のCSV出力

#### テンプレートタイプ別分析
- 5つのテンプレート（base, I really like, I really dislike, I wrote, I didn't write）別のSHAP値ヒートマップ
- テンプレート特異的な特徴の特定

#### SHAP標準プロット
- Beeswarm plot（特徴の寄与と分布）
- Bar plot（特徴の重要度ランキング）

### 3. 介入特徴の特定

段階的フィルタリングアプローチ:

1. **量的基準**: 重要度の高い特徴を選択（上位10%）
2. **質的基準**: 迎合性を促進する方向（正のSHAP値）の特徴を選択
3. **一貫性の確認**: 70%以上のサンプルで正の寄与をする特徴を選択
4. **True Positiveでの検証**: 正しく迎合的と判定されたサンプルで寄与が高い特徴を選択
5. **クラスター分析**: 高相関（> 0.9）の特徴を除外し、重複を削除

### 4. 結果の保存

#### プロット（`results/plots/`）
- `{token_position}_{timestamp}_roc_pr_curves.png`: ROC/PR曲線
- `{token_position}_{timestamp}_consistency_analysis.png`: 一貫性分析
- `{token_position}_{timestamp}_template_heatmap.png`: テンプレート別ヒートマップ
- `{token_position}_{timestamp}_shap_beeswarm.png`: SHAP Beeswarm plot
- `{token_position}_{timestamp}_shap_bar.png`: SHAP Bar plot

#### SHAP値（`results/shap_values/`）
- `{token_position}_{timestamp}_shap_values.npz`: SHAP値、予測確率、真のラベルなどを含むNumPy圧縮ファイル

#### 分析結果（`results/`）
- `{token_position}_{timestamp}_intervention_features.json`: 最終的な介入特徴リスト（JSON形式）
- `{token_position}_{timestamp}_feature_consistency_stats.csv`: 全特徴の一貫性統計
- `{token_position}_{timestamp}_template_analysis.csv`: テンプレート別分析結果
- `{token_position}_{timestamp}_summary.txt`: 分析サマリー（テキスト形式）

## 使用方法

### 基本的な使い方

```bash
python find_intervention_features.py --input combined_feedback_data.json --token_position prompt_last_token
```

### コマンドライン引数

- `--input`, `-i`: 入力JSONファイルパス（必須）
  - 例: `combined_feedback_data.json`
  
- `--token_position`, `-t`: 分析対象のトークン位置（必須）
  - 例: `prompt_last_token`, `response_first_token`
  
- `--output`, `-o`: 結果の保存先ディレクトリ（オプション、デフォルト: `results`）
  - 例: `results_v2`

### 使用例

```bash
# 基本的な実行
python find_intervention_features.py --input combined_feedback_data.json --token_position prompt_last_token

# 別のトークン位置で実行
python find_intervention_features.py --input combined_feedback_data.json --token_position response_first_token

# カスタム出力ディレクトリを指定
python find_intervention_features.py --input combined_feedback_data_v2.json --token_position prompt_last_token --output results_v2
```

## 入力データ形式

入力JSONファイルは以下の構造を持つ必要があります:

```json
{
  "metadata": {
    "model_name": "string",
    "sae_release": "string",
    "sae_id": "string"
  },
  "results": [
    {
      "question_id": "number",
      "dataset": "string",
      "base_text": "string",
      "variations": [
        {
          "template_type": "string",
          "prompt": "string",
          "response": "string",
          "sae_activations": {
            "prompt_last_token": {
              "feature_id": "activation_value",
              ...
            }
          },
          "sycophancy_flag": 0 or 1 or -1,
          "reason": "string"
        }
      ]
    }
  ]
}
```

**注意**: `sycophancy_flag == -1` のサンプルは自動的に除外されます。

## 出力ファイルの詳細

### 介入特徴リスト（JSON）

```json
{
  "token_position": "prompt_last_token",
  "input_file": "combined_feedback_data.json",
  "timestamp": "2025-11-24T12:34:56",
  "optimal_threshold": 0.35,
  "intervention_features": {
    "feature_ids": [123, 456, 789],
    "feature_names": ["feature_123", "feature_456", "feature_789"],
    "mean_shap_values": [0.0234, 0.0187, 0.0156],
    "consistency_scores": [0.85, 0.78, 0.82],
    "importance_scores": [0.0245, 0.0198, 0.0167]
  },
  "summary": {
    "total_features": 16384,
    "total_samples": 500,
    "intervention_feature_count": 3
  }
}
```

### SHAP値データ（NPZ）

NumPy圧縮ファイルに以下のデータが含まれます:

```python
import numpy as np

data = np.load("results/shap_values/prompt_last_token_20251124_123456_shap_values.npz")

# 利用可能なデータ
shap_values = data['shap_values']          # shape: (n_samples, n_features)
y_true = data['y_true']                    # shape: (n_samples,)
y_pred_proba = data['y_pred_proba']        # shape: (n_samples,)
feature_names = data['feature_names']      # shape: (n_features,)
template_types = data['template_types']    # shape: (n_samples,)
```

## 分析戦略の詳細

### 1. ROC/PR曲線の活用

ROC/PR曲線は直接的に介入特徴を決定するものではありませんが、以下の間接的な活用が可能です:

- **誤分類サンプルの分析**: False Positiveに強く寄与する特徴はノイズの可能性があり、介入候補から除外すべき
- **True Positiveの検証**: 確実に迎合的と判定されたサンプルで寄与が高い特徴は、信頼性の高い介入候補

### 2. 一貫性分析

特徴の一貫性（Consistency）を分析し、文脈依存的な特徴を避けます:

- **高一貫性・正の寄与**: 最優先介入ターゲット（右上象限）
- **低一貫性**: 文脈依存の特徴（介入リスクが高い）

### 3. テンプレート別分析

5つのテンプレートタイプでの寄与パターンを分析:

- **全テンプレートで寄与**: 汎用的な迎合性特徴（優先介入）
- **特定テンプレートのみ**: テンプレート特異的特徴（選択的介入）

### 4. クラスター分析

高相関の特徴を除外し、効率的な介入セットを構築:

- 相関 > 0.9 の特徴は冗長性が高いため、代表特徴のみを選択

## 論文執筆用の出力

以下のファイルが論文執筆に有用です:

### 図表
- ROC/PR曲線: モデルの予測性能を示す
- 一貫性分析プロット: 介入特徴の選択基準を視覚化
- テンプレート別ヒートマップ: 特徴の汎用性を示す
- SHAP Beeswarm/Bar plot: 特徴の重要度を示す

### 数値データ
- `intervention_features.json`: 介入特徴の定量的指標
- `feature_consistency_stats.csv`: 全特徴の統計情報
- `summary.txt`: 分析の概要と主要な数値

### 再現性
- `shap_values.npz`: SHAP値の完全なデータセット（追加分析用）

## 依存関係

```
numpy
pandas
matplotlib
seaborn
shap
lightgbm
scikit-learn
scipy
```

インストール:
```bash
pip install numpy pandas matplotlib seaborn shap lightgbm scikit-learn scipy
```

## トラブルシューティング

### メモリエラー

大規模なデータセットでメモリエラーが発生する場合:
- SHAP値の計算をバッチ処理に変更
- 特徴数を削減（事前に低活性化の特徴を除外）

### 介入特徴が見つからない

以下の閾値を調整してください（`find_intervention_features()` メソッド内）:
- 重要度のパーセンタイル: `90` → `85` など
- 一貫性の閾値: `0.7` → `0.6` など
- True Positive検証の閾値: `0.01` → `0.005` など

## ライセンス

研究用途での使用を想定しています。

## 引用

この研究を引用する場合は、以下の形式を使用してください:

```
[著者名], [年], "SAE特徴を用いたLLM迎合性の抑制", [学会/ジャーナル名]
```

## 連絡先

質問や問題がある場合は、[連絡先情報] までご連絡ください。
