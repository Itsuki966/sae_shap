# SAE特徴選択プログラム (feature_selection.py)

## 概要
`feature_selection.py`は、LightGBMとSHAP分析を用いて、迎合性（Sycophancy）に対する介入候補となるSAE特徴を選定するプログラムです。抑制候補（迎合性を促進する特徴）と増幅候補（非迎合性を促進する特徴）の両方を特定します。

## 処理の流れ

### 1. データ読み込み (`load_data`)
- ステップ2で作成されたJSONファイル（`combined_feedback_data.json`など）を読み込み
- 各サンプルの以下の情報を抽出:
  - `sae_activations`: SAE特徴の活性化値（指定されたトークン位置）
  - `sycophancy_flag`: 迎合性フラグ（0=非迎合、1=迎合）
  - `template_type`: テンプレートタイプ（base, I really like, I really dislike, I wrote, I didn't write）
- `sycophancy_flag == -1`のサンプルは除外
- 特徴量をDataFrame形式に変換（欠損値は0で埋める）

### 2. モデル学習とSHAP値計算 (`train_model_and_compute_shap`)
**キャッシュ機能:**
- `data/shap_values.npz`が存在する場合は再利用
- 存在しない場合は以下を実行:

**学習プロセス:**
1. **5-Fold Stratified Cross-Validation**でLightGBMモデルを学習
2. 各Foldで検証データに対してSHAP値を計算
3. 全Foldの結果を統合し、元のデータ順に整列
4. モデル性能（Accuracy, Precision, Recall, F1, ROC AUC）を表示
5. SHAP値をキャッシュファイルに保存

### 3. 指標計算 (`compute_metrics`)
各SAE特徴について、以下の指標を計算します（詳細は次セクション参照）:
- 基本統計量（活性化頻度、平均強度）
- 比較指標（Specificity, Consistency, Diff Base-Syc, Log Ratio, SHAP Correlation）
- 介入スコア（Suppression Score, Amplification Score）

### 4. 候補選定と保存 (`filter_and_save_candidates`)
- **抑制候補**: Suppression Scoreの上位20件を選定
  - 安全フィルタ: `Mean Intensity Base > 0.5`の特徴は除外
- **増幅候補**: Amplification Scoreの上位20件を選定
- 各候補リストをCSVファイルに保存

### 5. サマリー作成 (`save_summary`)
- 実験の設定、データ統計、上位10件の候補を含むサマリーファイルを生成

---

## 介入候補選定に使用される指標

### A. 基本統計量

#### 1. **Freq Syc (%)** - 迎合時の活性化頻度
- **計算方法**: 迎合サンプル（`flag=1`）のうち、当該特徴が活性化（値 > 0）している割合
- **意味**: この特徴が迎合的な応答でどれだけ頻繁に活性化するか

#### 2. **Freq NonSyc (%)** - 非迎合時の活性化頻度
- **計算方法**: 非迎合サンプル（`flag=0`）のうち、当該特徴が活性化している割合
- **意味**: この特徴が非迎合的な応答でどれだけ頻繁に活性化するか

#### 3. **Freq Base (%)** - Base時の活性化頻度
- **計算方法**: Baseテンプレートのサンプルのうち、当該特徴が活性化（値 > 0）している割合
- **意味**: ユーザーの意見が含まれない中立的なプロンプトでの活性化頻度（ベースライン頻度）

#### 4. **Mean Intensity Syc** - 迎合時の平均強度
- **計算方法**: 迎合サンプルにおける当該特徴の平均活性化値
- **意味**: 迎合時にこの特徴がどれだけ強く活性化するか

#### 5. **Mean Intensity NonSyc** - 非迎合時の平均強度
- **計算方法**: 非迎合サンプルにおける当該特徴の平均活性化値
- **意味**: 非迎合時にこの特徴がどれだけ強く活性化するか

#### 6. **Mean Intensity Base** - Base時の平均強度
- **計算方法**: **Baseテンプレートのサンプルのみ**における当該特徴の平均活性化値
- **意味**: ユーザーの意見が含まれない中立的なプロンプトでの活性化レベル（ベースライン）
- **重要性**: 介入の安全性を評価する基準値として使用

### B. 比較指標

#### 1. **Freq Diff Base-Syc** - Base-迎合の頻度差
- **計算方法**: `Freq Syc (%) - Freq Base (%)`
- **意味**: ベースラインと比較して、迎合時にどれだけ活性化頻度が変化するか
- **解釈**:
  - 正の値: 迎合時に頻度が増加（より頻繁に活性化）
  - 負の値: 迎合時に頻度が減少（活性化が抑えられる）
  - 0付近: ベースラインと頻度がほぼ同じ

#### 2. **Specificity** - 特異性
- **計算方法**: `Freq Syc / (Freq Syc + Freq NonSyc + ε)`
- **意味**: この特徴が迎合に対してどれだけ特異的か（1に近いほど迎合特異的）
- **解釈**:
  - 1.0に近い: ほぼ迎合時のみ活性化
  - 0.5付近: 迎合・非迎合の両方で均等に活性化
  - 0.0に近い: ほぼ非迎合時のみ活性化

#### 3. **Consistency** - 一貫性
- **計算方法**: `Freq Syc / 100`
- **意味**: 迎合サンプル内での活性化の一貫性（常に活性化するかどうか）
- **解釈**: 1.0に近いほど、迎合時に一貫して活性化

#### 4. **Diff Base-Syc** - Base-迎合の強度差
- **計算方法**: `Mean Intensity Syc - Mean Intensity Base`
- **意味**: ベースラインと比較して、迎合時にどれだけ活性化が増加するか
- **解釈**:
  - 正の値: 迎合時に活性化が増加（抑制候補）
  - 負の値: 迎合時に活性化が減少（増幅候補の可能性）

#### 5. **Log Ratio Syc/Base** - 迎合/Base の対数比
- **計算方法**: `log₂((Mean Intensity Syc + ε) / (Mean Intensity Base + ε))`
- **意味**: ベースラインに対する迎合時の活性化の倍率（対数スケール）
- **解釈**:
  - 1.0: 迎合時に2倍活性化
  - 2.0: 迎合時に4倍活性化
  - -1.0: 迎合時に半減

#### 6. **SHAP Correlation** - SHAP相関
- **計算方法**: 特徴量値とSHAP値の相関係数
- **意味**: この特徴の活性化が迎合性予測に正の寄与をするか、負の寄与をするか
- **解釈**:
  - 正の値: 活性化が増えると迎合予測が増加
  - 負の値: 活性化が増えると迎合予測が減少

### C. 介入スコア

#### 1. **Suppression Score** - 抑制スコア
**対象**: SHAP Correlation > 0（迎合を促進する特徴）

**計算方法**:
```
freq_gain = max(0, Freq Diff Base-Syc)
Suppression Score = Specificity × Consistency × max(0, Diff Base-Syc) × max(0, Log Ratio Syc/Base) × (freq_gain / 100.0 + 1.0)
```

**各要素の役割**:
- `Specificity`: 迎合に特異的か
- `Consistency`: 迎合時に一貫して活性化するか
- `max(0, Diff Base-Syc)`: ベースラインより強度が増加しているか
- `max(0, Log Ratio Syc/Base)`: 増加が統計的に有意か
- `(freq_gain / 100.0 + 1.0)`: 頻度ボーナス（頻度が増加した特徴を優遇）

**意味**: この特徴を抑制（活性化を0にする）することで、迎合性を効果的に減少させられる可能性が高いかを示す総合スコア

**高スコアの特徴の特性**:
- 迎合時にほぼ常に活性化
- 非迎合時にはあまり活性化しない
- ベースラインより大幅に活性化強度が増加
- ベースラインより活性化頻度も増加（頻度ボーナスにより評価が高まる）
- SHAP分析で迎合性予測に正の寄与

#### 2. **Amplification Score** - 増幅スコア
**対象**: SHAP Correlation < 0（非迎合を促進する特徴）

**計算方法**:
```
vanished = max(0, Mean Intensity Base - Mean Intensity Syc)
Amplification Score = (1 - Specificity) × (Freq NonSyc / 100) × vanished
```

**各要素の役割**:
- `(1 - Specificity)`: 非迎合に特異的か
- `(Freq NonSyc / 100)`: 非迎合時の一貫性
- `vanished`: 迎合時に失われた活性化量（ベースラインとの差）

**意味**: この特徴を増幅（活性化を強制的に増やす）することで、迎合性を効果的に減少させられる可能性が高いかを示す総合スコア

**高スコアの特徴の特性**:
- 非迎合時に頻繁に活性化
- 迎合時には活性化が減少（ベースラインより低い）
- SHAP分析で迎合性予測に負の寄与

---

## 出力ファイル

### ディレクトリ構造
```
results/
└── experiments/
    └── feature_selection_{token_position}_{timestamp}/
        ├── data/
        │   ├── shap_values.npz
        │   ├── feature_metrics_full.csv
        │   ├── candidates_suppress.csv
        │   └── candidates_amplify.csv
        ├── figures/
        └── summary.txt
```

### 1. `data/shap_values.npz`
**内容**:
- `shap_values`: 全サンプル×全特徴のSHAP値配列
- `feature_names`: 特徴名リスト
- `y_true`: 真のラベル
- `y_pred_proba`: 予測確率

**保存場所**: `results/experiments/feature_selection_{token_position}_{timestamp}/data/`

**活用方法**:
- 次回実行時のキャッシュとして使用（再計算をスキップ）
- 詳細なSHAP分析や可視化に使用
- 他の分析プログラムへの入力データとして使用

### 2. `data/feature_metrics_full.csv`
**内容**: 全SAE特徴の全指標を含む完全なデータセット

**列**:
- `Feature`: 特徴名（例: feature_1234）
- `Feature_ID`: 特徴ID（数値）
- `Freq Syc (%)`: 迎合時の活性化頻度
- `Freq NonSyc (%)`: 非迎合時の活性化頻度
- `Freq Base (%)`: Base時の活性化頻度
- `Freq Diff Base-Syc`: Base-迎合の頻度差
- `Mean Intensity Syc`: 迎合時の平均強度
- `Mean Intensity NonSyc`: 非迎合時の平均強度
- `Mean Intensity Base`: Base時の平均強度
- `Specificity`: 特異性
- `Consistency`: 一貫性
- `Diff Base-Syc`: Base-迎合の強度差
- `Log Ratio Syc/Base`: 迎合/Baseの対数比
- `SHAP Correlation`: SHAP相関
- `Suppression Score`: 抑制スコア
- `Amplification Score`: 増幅スコア

**保存場所**: `results/experiments/feature_selection_{token_position}_{timestamp}/data/`

**活用方法**:
- 全特徴の包括的な分析
- カスタムフィルタリング条件の適用
- 追加の統計分析や可視化
- 異なる閾値での候補再選定

### 3. `data/candidates_suppress.csv`
**内容**: Suppression Score上位20件の抑制候補特徴

**フィルタ条件**:
- `Mean Intensity Base ≤ 0.5`（安全フィルタ: ベースラインで強く活性化する特徴は除外）
- Suppression Score降順でソート

**保存場所**: `results/experiments/feature_selection_{token_position}_{timestamp}/data/`

**活用方法**:
- **ステップ4（介入実験）の入力**: これらの特徴IDを使用してAblation（抑制）介入を実行
- 介入の優先順位付け（上位ほど効果が高い可能性）
- 安全性評価（Mean Intensity Baseが低いことを確認）
- 複数の特徴を組み合わせた介入戦略の設計

### 4. `data/candidates_amplify.csv`
**内容**: Amplification Score上位20件の増幅候補特徴

**フィルタ条件**:
- Amplification Score降順でソート

**保存場所**: `results/experiments/feature_selection_{token_position}_{timestamp}/data/`

**活用方法**:
- **ステップ4（介入実験）の入力**: これらの特徴IDを使用して増幅介入を実行
- 抑制とは異なるアプローチでの迎合性緩和実験
- 非迎合性を促進するメカニズムの理解
- 抑制候補と組み合わせたハイブリッド介入戦略

### 5. `summary.txt`
**内容**:
- 実験の設定情報（入力ファイル、トークン位置、実行日時）
- データ統計（特徴数、サンプル数、クラス分布）
- 抑制候補上位10件の詳細
- 増幅候補上位10件の詳細

**保存場所**: `results/experiments/feature_selection_{token_position}_{timestamp}/`

**活用方法**:
- 実験結果のクイックレビュー
- 実験ログとしての記録
- 複数実験の結果比較
- 論文・レポート用のサマリー

---

## 使用方法

### 基本的な使用
```bash
python feature_selection.py --input combined_feedback_data.json --token_position prompt_last_token
```

### SHAP値の強制再計算
```bash
python feature_selection.py --input combined_feedback_data.json --token_position prompt_last_token --force-recompute
```

### カスタム出力ディレクトリ
```bash
python feature_selection.py --input combined_feedback_data_v2.json --token_position response_first_token --output results_v2
```

### コマンドライン引数
- `--input`, `-i`: 入力JSONファイルパス（デフォルト: `combined_feedback_data.json`）
- `--token_position`, `-t`: 分析対象のトークン位置（必須）
- `--output`, `-o`: 結果の保存先ディレクトリ（デフォルト: `results`）
- `--force-recompute`: キャッシュを無視してSHAP値を再計算

---

## 注意事項

### 1. 安全フィルタの重要性
抑制候補の選定時、`Mean Intensity Base > 0.5`の特徴は除外されます。これは以下の理由によります:
- ベースラインで強く活性化する特徴は、モデルの基本的な機能に関与している可能性が高い
- これらを抑制すると、迎合性以外の重要な能力も損なわれるリスクがある
- 安全性を優先し、ベースラインでの活性化が低い特徴のみを候補とする

### 2. SHAP相関の解釈
- SHAP Correlation > 0: 特徴の活性化が迎合性を促進 → 抑制候補
- SHAP Correlation < 0: 特徴の活性化が迎合性を抑制 → 増幅候補
- この分類により、介入の方向性（抑制 vs 増幅）が決定される

### 3. スコアの相対的評価
- Suppression ScoreとAmplification Scoreは異なる式で計算されるため、直接比較できない
- 各スコア内での相対的な順位が重要
- 実際の介入効果は、ステップ4の介入実験で検証する必要がある

### 4. キャッシュの利用
- 初回実行後、SHAP値がキャッシュされるため、2回目以降は高速に実行可能
- データやモデルパラメータを変更した場合は`--force-recompute`で再計算を推奨
- 異なるトークン位置を分析する場合は、自動的に別のディレクトリに保存される

---

## 次のステップ

このプログラムで選定された介入候補特徴は、**ステップ4: 介入と出力操作**で使用されます:

1. `candidates_suppress.csv`から特徴IDを抽出
2. `HookedTransformer`のフック機能を使用して、これらの特徴を抑制（活性化を0にする）
3. 迎合的なプロンプトで応答を生成し、介入の効果を評価
4. 必要に応じて`candidates_amplify.csv`の特徴も使用した増幅介入を実施

介入実験の詳細は、別途作成される介入実験用スクリプトを参照してください。
