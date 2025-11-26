#!/usr/bin/env python3
"""
SAE特徴の分析と介入候補選定スクリプト

このプログラムは、LightGBMとSHAP分析を用いて、迎合性（Sycophancy）に対する
介入候補特徴を選定します。抑制候補と増幅候補の両方を特定します。

使用方法:
    python feature_selection.py --input combined_feedback_data.json --token_position prompt_last_token
"""

import argparse
import json
import os
from pathlib import Path
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
)

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Hiragino Sans'  # macOS
plt.rcParams['font.size'] = 10


class FeatureSelector:
    """SAE特徴の介入候補を選定するクラス"""
    
    def __init__(self, input_file, token_position, output_dir="results"):
        """
        Args:
            input_file: 入力JSONファイルパス
            token_position: 分析対象のトークン位置 (例: "prompt_last_token")
            output_dir: 結果の保存先ディレクトリ
        """
        self.input_file = input_file
        self.token_position = token_position
        
        # タイムスタンプ付きの実験ディレクトリ
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"feature_selection_{token_position}_{timestamp}"
        
        # ディレクトリ構造の設定
        self.output_dir = Path(output_dir)
        self.experiment_dir = self.output_dir / "experiments" / experiment_name
        self.data_dir = self.experiment_dir / "data"
        self.figures_dir = self.experiment_dir / "figures"
        
        # 出力ディレクトリの作成
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.file_prefix = experiment_name
        self.timestamp = timestamp
        
        # データ格納用
        self.data = None
        self.X = None
        self.y = None
        self.template_types = None
        self.shap_values = None
        self.feature_names = None
        self.y_true = None
        self.y_pred_proba = None
        
        # モデル性能指標
        self.model_metrics = {}
        self.roc_auc = None
        self.avg_precision = None
        self.optimal_threshold = None
        
        # キャッシュファイルのパス
        self.shap_cache_path = self.data_dir / "shap_values.npz"
        
        print(f"=== SAE特徴選択プログラム ===")
        print(f"入力ファイル: {input_file}")
        print(f"トークン位置: {token_position}")
        print(f"出力ディレクトリ: {self.experiment_dir}\n")
    
    def load_data(self):
        """JSONファイルからデータを読み込む"""
        print("=== データ読み込み ===")
        with open(self.input_file, "r") as f:
            self.data = json.load(f)
        
        feedback_results = self.data["results"]
        print(f"読み込み完了: {len(feedback_results)} 件の質問データ")
        
        # SAE特徴とフラグを抽出
        self.X, self.y, self.template_types = self._extract_features_and_labels(feedback_results)
        
        print(f"特徴数: {self.X.shape[1]}")
        print(f"サンプル数: {self.X.shape[0]}")
        print(f"クラス分布: Flag=0: {(self.y == 0).sum()}, Flag=1: {(self.y == 1).sum()}")
        print(f"テンプレートタイプ: {np.unique(self.template_types)}\n")
        
        return self.X, self.y
    
    def _extract_features_and_labels(self, feedback_results):
        """フィードバック結果からSAE特徴とラベルを抽出"""
        activations_list = []
        y_list = []
        template_types = []
        skipped_count = 0
        
        for result in feedback_results:
            for variation in result["variations"]:
                flag = variation["sycophancy_flag"]
                
                # フラグが -1 のデータは除外
                if flag == -1:
                    skipped_count += 1
                    continue
                
                y_list.append(flag)
                template_types.append(variation["template_type"])
                
                # 指定されたトークン位置の特徴量を取得
                activations = variation["sae_activations"].get(self.token_position, {})
                activations_list.append(activations)
        
        print(f"除外されたサンプル (flag == -1): {skipped_count}")
        
        # DataFrameに変換
        X = pd.DataFrame(activations_list).fillna(0)
        X.columns = X.columns.astype(int)
        X = X.sort_index(axis=1)
        X.columns = [f"feature_{col}" for col in X.columns]
        
        return X, np.array(y_list), np.array(template_types)
    
    def train_model_and_compute_shap(self, n_splits=5, force_recompute=False):
        """モデルを学習しSHAP値を計算（キャッシュ機能付き）"""
        
        # キャッシュが存在し、再計算不要な場合はロード
        if self.shap_cache_path.exists() and not force_recompute:
            print("=== キャッシュからSHAP値を読み込み ===")
            cached_data = np.load(self.shap_cache_path, allow_pickle=True)
            self.shap_values = cached_data['shap_values']
            self.feature_names = cached_data['feature_names']
            self.y_true = cached_data['y_true']
            self.y_pred_proba = cached_data['y_pred_proba']
            print(f"SHAP値を読み込み: {self.shap_values.shape}")
            print(f"特徴名数: {len(self.feature_names)}\n")
            return
        
        print("=== モデル学習とSHAP値計算 ===")
        
        # データ型変換
        X = self.X.astype(float)
        y = self.y.astype(int)
        
        # Stratified K-Fold クロスバリデーション
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # LightGBMパラメータ
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'min_data_in_leaf': 20,
            'min_sum_hessian_in_leaf': 1e-3,
            'max_depth': -1
        }
        
        # 予測確率とSHAP値を収集
        all_y_true = []
        all_y_pred_proba = []
        shap_values_list = []
        indices_list = []
        
        print(f"{n_splits}-Fold クロスバリデーション開始...")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"  Fold {fold}/{n_splits}...", end=" ")
            
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
            val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)
            
            model_fold = lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
            )
            
            y_pred_proba = model_fold.predict(X_val_fold, num_iteration=model_fold.best_iteration)
            all_y_true.extend(y_val_fold)
            all_y_pred_proba.extend(y_pred_proba)
            
            # SHAP値の計算
            explainer_fold = shap.TreeExplainer(model_fold)
            shap_explanation = explainer_fold(X_val_fold)
            shap_values_fold = shap_explanation.values
            
            if isinstance(shap_values_fold, list):
                shap_values_fold = shap_values_fold[1]
            
            shap_values_list.append(shap_values_fold)
            indices_list.append(val_idx)
            
            # Fold性能
            y_pred_fold = (y_pred_proba >= 0.5).astype(int)
            acc = accuracy_score(y_val_fold, y_pred_fold)
            f1 = f1_score(y_val_fold, y_pred_fold, zero_division=0)
            print(f"Acc: {acc:.4f}, F1: {f1:.4f}")
        
        # SHAP値の整列
        all_shap_values = np.concatenate(shap_values_list, axis=0)
        all_indices = np.concatenate(indices_list, axis=0)
        
        sorted_idx = np.argsort(all_indices)
        self.shap_values = all_shap_values[sorted_idx]
        self.feature_names = self.X.columns.to_numpy()
        
        # 全体の性能を表示
        self.y_true = np.array(all_y_true)[sorted_idx]
        self.y_pred_proba = np.array(all_y_pred_proba)[sorted_idx]
        y_pred = (self.y_pred_proba >= 0.5).astype(int)
        
        print(f"\n=== 全体性能 ===")
        print(f"Accuracy:  {accuracy_score(self.y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(self.y_true, y_pred, zero_division=0):.4f}")
        print(f"Recall:    {recall_score(self.y_true, y_pred, zero_division=0):.4f}")
        print(f"F1 Score:  {f1_score(self.y_true, y_pred, zero_division=0):.4f}")
        print(f"ROC AUC:   {roc_auc_score(self.y_true, self.y_pred_proba):.4f}\n")
        
        # キャッシュに保存
        np.savez(
            self.shap_cache_path,
            shap_values=self.shap_values,
            feature_names=self.feature_names,
            y_true=self.y_true,
            y_pred_proba=self.y_pred_proba
        )
        print(f"SHAP値をキャッシュに保存: {self.shap_cache_path}\n")
    
    def compute_metrics(self):
        """特徴ごとの指標を計算"""
        print("=== 指標計算 ===")
        
        eps = 1e-6  # ゼロ除算回避
        
        # マスクの作成
        mask_syc = self.y == 1
        mask_nonsyc = self.y == 0
        mask_base = self.template_types == "base"
        
        print(f"迎合サンプル: {mask_syc.sum()}")
        print(f"非迎合サンプル: {mask_nonsyc.sum()}")
        print(f"Baseテンプレート: {mask_base.sum()}\n")
        
        metrics_list = []
        
        for i, feature_name in enumerate(self.feature_names):
            feature_vals = self.X[feature_name].values
            feature_shap = self.shap_values[:, i]
            
            # 1. 基本統計量
            # 活性化閾値（0より大きい）
            active_syc = feature_vals[mask_syc] > 0
            active_nonsyc = feature_vals[mask_nonsyc] > 0
            active_base = feature_vals[mask_base] > 0
            
            freq_syc = (active_syc.sum() / len(active_syc) * 100) if mask_syc.sum() > 0 else 0
            freq_nonsyc = (active_nonsyc.sum() / len(active_nonsyc) * 100) if mask_nonsyc.sum() > 0 else 0
            
            mean_intensity_syc = feature_vals[mask_syc].mean() if mask_syc.sum() > 0 else 0
            mean_intensity_nonsyc = feature_vals[mask_nonsyc].mean() if mask_nonsyc.sum() > 0 else 0
            mean_intensity_base = feature_vals[mask_base].mean() if mask_base.sum() > 0 else 0
            
            # 2. 比較指標
            specificity = freq_syc / (freq_syc + freq_nonsyc + eps)
            consistency = freq_syc / 100.0
            diff_base_syc = mean_intensity_syc - mean_intensity_base
            log_ratio_syc_base = np.log2((mean_intensity_syc + eps) / (mean_intensity_base + eps))
            
            # SHAP相関（特徴量値とSHAP値の相関）
            if len(feature_vals) > 1 and feature_vals.std() > 0:
                shap_correlation = np.corrcoef(feature_vals, feature_shap)[0, 1]
            else:
                shap_correlation = 0
            
            # 3. スコア算出
            # Suppression Score (抑制): SHAP Correlation > 0 の場合のみ
            if shap_correlation > 0:
                suppression_score = (
                    specificity * 
                    consistency * 
                    max(0, diff_base_syc) * 
                    max(0, log_ratio_syc_base)
                )
            else:
                suppression_score = 0
            
            # Amplification Score (増幅): SHAP Correlation < 0 の場合のみ
            if shap_correlation < 0:
                vanished = max(0, mean_intensity_base - mean_intensity_syc)
                amplification_score = (
                    (1 - specificity) * 
                    (freq_nonsyc / 100.0) * 
                    vanished
                )
            else:
                amplification_score = 0
            
            metrics_list.append({
                'Feature': feature_name,
                'Feature_ID': int(feature_name.replace('feature_', '')),
                'Freq Syc (%)': freq_syc,
                'Freq NonSyc (%)': freq_nonsyc,
                'Mean Intensity Syc': mean_intensity_syc,
                'Mean Intensity NonSyc': mean_intensity_nonsyc,
                'Mean Intensity Base': mean_intensity_base,
                'Specificity': specificity,
                'Consistency': consistency,
                'Diff Base-Syc': diff_base_syc,
                'Log Ratio Syc/Base': log_ratio_syc_base,
                'SHAP Correlation': shap_correlation,
                'Suppression Score': suppression_score,
                'Amplification Score': amplification_score
            })
        
        df_metrics = pd.DataFrame(metrics_list)
        
        print(f"指標計算完了: {len(df_metrics)} 特徴\n")
        
        return df_metrics
    
    def analyze_roc_pr_curves(self):
        """ROC曲線とPR曲線を分析"""
        print("=== ROC/PR曲線の分析 ===")
        
        # ROC曲線
        fpr, tpr, thresholds_roc = roc_curve(self.y_true, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # PR曲線
        precision, recall, thresholds_pr = precision_recall_curve(
            self.y_true, self.y_pred_proba
        )
        avg_precision = average_precision_score(self.y_true, self.y_pred_proba)
        
        # メトリクスを保存
        self.roc_auc = roc_auc
        self.avg_precision = avg_precision
        
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        
        # 可視化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # ROC曲線
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate', fontsize=12)
        ax1.set_ylabel('True Positive Rate', fontsize=12)
        ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax1.legend(loc="lower right")
        ax1.grid(alpha=0.3)
        
        # PR曲線
        ax2.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
        ax2.axhline(
            y=self.y_true.mean(), color='navy', linestyle='--', lw=2,
            label=f'Baseline ({self.y_true.mean():.3f})'
        )
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall', fontsize=12)
        ax2.set_ylabel('Precision', fontsize=12)
        ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax2.legend(loc="lower left")
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "model_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 最適閾値の探索
        self._find_optimal_threshold(thresholds_roc, tpr, fpr)
        
        print(f"ROC/PR曲線を保存しました\n")
    
    def _find_optimal_threshold(self, thresholds_roc, tpr, fpr):
        """最適閾値を探索"""
        print("\n=== 最適閾値の探索 ===")
        
        # F1スコアが最大になる閾値
        thresholds_test = np.arange(0.1, 0.9, 0.05)
        f1_scores = []
        
        for threshold in thresholds_test:
            y_pred = (self.y_pred_proba >= threshold).astype(int)
            f1 = f1_score(self.y_true, y_pred, zero_division=0)
            f1_scores.append(f1)
        
        optimal_threshold_f1 = thresholds_test[np.argmax(f1_scores)]
        max_f1 = np.max(f1_scores)
        
        self.optimal_threshold = optimal_threshold_f1
        
        y_pred_optimal = (self.y_pred_proba >= optimal_threshold_f1).astype(int)
        acc_optimal = accuracy_score(self.y_true, y_pred_optimal)
        prec_optimal = precision_score(self.y_true, y_pred_optimal, zero_division=0)
        rec_optimal = recall_score(self.y_true, y_pred_optimal, zero_division=0)
        
        # 混同行列を計算
        cm = confusion_matrix(self.y_true, y_pred_optimal)
        
        # モデルメトリクスを保存
        self.model_metrics = {
            'optimal_threshold': float(optimal_threshold_f1),
            'accuracy': float(acc_optimal),
            'precision': float(prec_optimal),
            'recall': float(rec_optimal),
            'f1_score': float(max_f1),
            'roc_auc': float(self.roc_auc),
            'avg_precision': float(self.avg_precision),
            'confusion_matrix': cm.tolist(),
            'true_negatives': int(cm[0, 0]),
            'false_positives': int(cm[0, 1]),
            'false_negatives': int(cm[1, 0]),
            'true_positives': int(cm[1, 1])
        }
        
        print(f"F1スコア最大化:")
        print(f"  最適閾値: {optimal_threshold_f1:.2f}")
        print(f"  F1スコア: {max_f1:.4f}")
        print(f"  Accuracy: {acc_optimal:.4f}")
        print(f"  Precision: {prec_optimal:.4f}")
        print(f"  Recall: {rec_optimal:.4f}")
        print(f"\n混同行列:")
        print(f"  TN: {cm[0, 0]}, FP: {cm[0, 1]}")
        print(f"  FN: {cm[1, 0]}, TP: {cm[1, 1]}")
        
        # Youden's Index
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_threshold_youden = thresholds_roc[optimal_idx]
        
        print(f"\nYouden's Index:")
        print(f"  最適閾値: {optimal_threshold_youden:.4f}\n")
    
    def filter_and_save_candidates(self, df_metrics):
        """候補特徴をフィルタリングして保存"""
        print("=== 候補特徴の選定 ===")
        
        # 1. 全特徴の指標を保存
        full_metrics_path = self.data_dir / "feature_metrics_full.csv"
        df_metrics.to_csv(full_metrics_path, index=False)
        print(f"全特徴の指標を保存: {full_metrics_path}")
        
        # 2. 抑制候補の選定
        # 安全フィルタ: Mean Intensity Base > 0.5 は除外
        df_suppress = df_metrics[df_metrics['Mean Intensity Base'] <= 0.5].copy()
        df_suppress = df_suppress.sort_values('Suppression Score', ascending=False)
        df_suppress_top20 = df_suppress.head(20)
        
        suppress_path = self.data_dir / "candidates_suppress.csv"
        df_suppress_top20.to_csv(suppress_path, index=False)
        print(f"抑制候補 (上位20件) を保存: {suppress_path}")
        print(f"  - 候補数: {len(df_suppress_top20)}")
        if len(df_suppress_top20) > 0:
            print(f"  - Top 1: {df_suppress_top20.iloc[0]['Feature']} (Score: {df_suppress_top20.iloc[0]['Suppression Score']:.6f})")
        
        # 3. 増幅候補の選定
        df_amplify = df_metrics.sort_values('Amplification Score', ascending=False)
        df_amplify_top20 = df_amplify.head(20)
        
        amplify_path = self.data_dir / "candidates_amplify.csv"
        df_amplify_top20.to_csv(amplify_path, index=False)
        print(f"増幅候補 (上位20件) を保存: {amplify_path}")
        print(f"  - 候補数: {len(df_amplify_top20)}")
        if len(df_amplify_top20) > 0:
            print(f"  - Top 1: {df_amplify_top20.iloc[0]['Feature']} (Score: {df_amplify_top20.iloc[0]['Amplification Score']:.6f})")
        
        print()
        
        return df_suppress_top20, df_amplify_top20
    
    def save_summary(self, df_suppress, df_amplify):
        """サマリーファイルを保存"""
        print("=== サマリー作成 ===")
        
        summary_path = self.experiment_dir / "summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write(f"=== SAE特徴選択サマリー ===\n\n")
            f.write(f"実行日時: {datetime.now().isoformat()}\n")
            f.write(f"入力ファイル: {self.input_file}\n")
            f.write(f"トークン位置: {self.token_position}\n\n")
            
            f.write(f"=== データ統計 ===\n")
            f.write(f"総特徴数: {self.X.shape[1]}\n")
            f.write(f"総サンプル数: {self.X.shape[0]}\n")
            f.write(f"迎合サンプル: {(self.y == 1).sum()}\n")
            f.write(f"非迎合サンプル: {(self.y == 0).sum()}\n")
            f.write(f"Baseテンプレート: {(self.template_types == 'base').sum()}\n\n")
            
            # モデル性能指標を追加
            if self.model_metrics:
                f.write(f"=== モデル性能 ===\n")
                f.write(f"ROC AUC: {self.model_metrics.get('roc_auc', 'N/A'):.4f}\n")
                f.write(f"Average Precision: {self.model_metrics.get('avg_precision', 'N/A'):.4f}\n")
                f.write(f"最適閾値: {self.model_metrics.get('optimal_threshold', 'N/A'):.2f}\n")
                f.write(f"Accuracy: {self.model_metrics.get('accuracy', 'N/A'):.4f}\n")
                f.write(f"Precision: {self.model_metrics.get('precision', 'N/A'):.4f}\n")
                f.write(f"Recall: {self.model_metrics.get('recall', 'N/A'):.4f}\n")
                f.write(f"F1 Score: {self.model_metrics.get('f1_score', 'N/A'):.4f}\n")
                f.write(f"\n混同行列:\n")
                f.write(f"  TN: {self.model_metrics.get('true_negatives', 'N/A')}, ")
                f.write(f"FP: {self.model_metrics.get('false_positives', 'N/A')}\n")
                f.write(f"  FN: {self.model_metrics.get('false_negatives', 'N/A')}, ")
                f.write(f"TP: {self.model_metrics.get('true_positives', 'N/A')}\n\n")
            
            f.write(f"=== 抑制候補 (上位10件) ===\n")
            for i, row in df_suppress.head(10).iterrows():
                f.write(f"{row['Feature']} (ID: {row['Feature_ID']})\n")
                f.write(f"  Suppression Score: {row['Suppression Score']:.6f}\n")
                f.write(f"  Specificity: {row['Specificity']:.4f}\n")
                f.write(f"  Consistency: {row['Consistency']:.4f}\n")
                f.write(f"  Diff Base-Syc: {row['Diff Base-Syc']:.6f}\n")
                f.write(f"  Mean Intensity Base: {row['Mean Intensity Base']:.6f}\n\n")
            
            f.write(f"=== 増幅候補 (上位10件) ===\n")
            for i, row in df_amplify.head(10).iterrows():
                f.write(f"{row['Feature']} (ID: {row['Feature_ID']})\n")
                f.write(f"  Amplification Score: {row['Amplification Score']:.6f}\n")
                f.write(f"  Specificity: {row['Specificity']:.4f}\n")
                f.write(f"  Freq NonSyc: {row['Freq NonSyc (%)']:.2f}%\n")
                f.write(f"  Mean Intensity Base: {row['Mean Intensity Base']:.6f}\n")
                f.write(f"  Mean Intensity Syc: {row['Mean Intensity Syc']:.6f}\n\n")
        
        print(f"サマリーを保存: {summary_path}\n")
    
    def run_full_pipeline(self):
        """完全なパイプラインを実行"""
        print("=" * 60)
        print("SAE特徴選択プログラム - 完全パイプライン")
        print("=" * 60)
        print()
        
        # 1. データ読み込み
        self.load_data()
        
        # 2. モデル学習とSHAP計算（キャッシュ機能付き）
        self.train_model_and_compute_shap()
        
        # 3. ROC/PR曲線分析と最適閾値探索
        self.analyze_roc_pr_curves()
        
        # 4. 指標計算
        df_metrics = self.compute_metrics()
        
        # 4. 候補選定と保存
        df_suppress, df_amplify = self.filter_and_save_candidates(df_metrics)
        
        # 5. サマリー保存
        self.save_summary(df_suppress, df_amplify)
        
        print("=" * 60)
        print("処理完了！")
        print(f"実験ディレクトリ: {self.experiment_dir}")
        print(f"  - データ: {self.data_dir}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="SAE特徴の分析と介入候補選定スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python feature_selection.py --input combined_feedback_data.json --token_position prompt_last_token
  python feature_selection.py --input combined_feedback_data_v2.json --token_position response_first_token --output results_v2
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='combined_feedback_data.json',
        help='入力JSONファイルパス（例: combined_feedback_data.json）'
    )
    
    parser.add_argument(
        '--token_position', '-t',
        type=str,
        required=True,
        help='分析対象のトークン位置（例: prompt_last_token, response_first_token）'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results',
        help='結果の保存先ディレクトリ（デフォルト: results）'
    )
    
    parser.add_argument(
        '--force-recompute',
        action='store_true',
        help='キャッシュを無視してSHAP値を再計算する'
    )
    
    args = parser.parse_args()
    
    # 分析実行
    selector = FeatureSelector(
        input_file=args.input,
        token_position=args.token_position,
        output_dir=args.output
    )
    
    selector.run_full_pipeline()


if __name__ == "__main__":
    main()
