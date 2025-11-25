#!/usr/bin/env python3
"""
SAE特徴の介入ターゲット特定プログラム

このプログラムは、迎合性（Sycophancy）抑制のために介入すべきSAE特徴を特定します。
SHAP値分析、機械学習モデル、複数の可視化手法を組み合わせて、効果的な介入ターゲットを発見します。

使用方法:
    python find_intervention_features.py --input combined_feedback_data.json --token_position prompt_last_token
"""

import argparse
import json
import os
from pathlib import Path
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

warnings.filterwarnings('ignore')
plt.rcParams['font.size'] = 10
sns.set_palette("husl")
plt.rcParams['font.family'] = 'Hiragino Sans'  # macOS
plt.rcParams['font.size'] = 10

class InterventionFeatureFinder:
    """介入特徴を特定するためのメインクラス"""
    
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
        experiment_name = f"{token_position}_{timestamp}"
        
        # ハイブリッド構造のディレクトリ設定
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
        self.shap_values_sorted = None
        self.y_true_sorted = None
        self.y_pred_proba_sorted = None
        self.optimal_threshold = None
        self.intervention_features = None
        
        # モデル性能指標の格納用
        self.model_metrics = {}
        self.cv_metrics = []
        self.lgb_params = None
        self.roc_auc = None
        self.avg_precision = None
        
        print(f"=== 介入特徴探索プログラム ===")
        print(f"入力ファイル: {input_file}")
        print(f"トークン位置: {token_position}")
        print(f"出力ディレクトリ: {output_dir}")
        print(f"ファイルプレフィックス: {self.file_prefix}\n")
    
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
        print(f"クラス分布: Flag=0: {(self.y == 0).sum()}, Flag=1: {(self.y == 1).sum()}\n")
        
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
    
    def train_model_and_compute_shap(self, n_splits=5):
        """モデルを学習しSHAP値を計算"""
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
        
        # パラメータを保存
        self.lgb_params = params.copy()
        self.lgb_params['n_splits'] = n_splits
        self.lgb_params['random_state'] = 42
        
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
            
            # 各Foldの性能を計算して保存
            y_pred_fold = (y_pred_proba >= 0.5).astype(int)
            fold_metrics = {
                'fold': fold,
                'accuracy': accuracy_score(y_val_fold, y_pred_fold),
                'precision': precision_score(y_val_fold, y_pred_fold, zero_division=0),
                'recall': recall_score(y_val_fold, y_pred_fold, zero_division=0),
                'f1': f1_score(y_val_fold, y_pred_fold, zero_division=0),
                'roc_auc': roc_curve(y_val_fold, y_pred_proba)[0:2],  # fpr, tpr
                'best_iteration': model_fold.best_iteration
            }
            # ROC AUCを計算
            fpr_fold, tpr_fold, _ = roc_curve(y_val_fold, y_pred_proba)
            fold_metrics['roc_auc'] = auc(fpr_fold, tpr_fold)
            fold_metrics['avg_precision'] = average_precision_score(y_val_fold, y_pred_proba)
            self.cv_metrics.append(fold_metrics)
            
            # SHAP値の計算
            explainer_fold = shap.TreeExplainer(model_fold)
            shap_explanation = explainer_fold(X_val_fold)
            shap_values_fold = shap_explanation.values
            
            if isinstance(shap_values_fold, list):
                shap_values_fold = shap_values_fold[1]
            
            shap_values_list.append(shap_values_fold)
            indices_list.append(val_idx)
            
            print("完了")
        
        # SHAP値の整列
        all_shap_values = np.concatenate(shap_values_list, axis=0)
        all_indices = np.concatenate(indices_list, axis=0)
        
        sorted_idx = np.argsort(all_indices)
        self.shap_values_sorted = all_shap_values[sorted_idx]
        self.y_true_sorted = np.array(all_y_true)[sorted_idx]
        self.y_pred_proba_sorted = np.array(all_y_pred_proba)[sorted_idx]
        
        print(f"SHAP値計算完了: {self.shap_values_sorted.shape}\n")
        
        # クロスバリデーション結果のサマリーを表示
        print("=== クロスバリデーション結果 ===")
        cv_acc = [m['accuracy'] for m in self.cv_metrics]
        cv_prec = [m['precision'] for m in self.cv_metrics]
        cv_rec = [m['recall'] for m in self.cv_metrics]
        cv_f1 = [m['f1'] for m in self.cv_metrics]
        cv_roc_auc = [m['roc_auc'] for m in self.cv_metrics]
        cv_avg_prec = [m['avg_precision'] for m in self.cv_metrics]
        
        print(f"Accuracy:  {np.mean(cv_acc):.4f} ± {np.std(cv_acc):.4f}")
        print(f"Precision: {np.mean(cv_prec):.4f} ± {np.std(cv_prec):.4f}")
        print(f"Recall:    {np.mean(cv_rec):.4f} ± {np.std(cv_rec):.4f}")
        print(f"F1 Score:  {np.mean(cv_f1):.4f} ± {np.std(cv_f1):.4f}")
        print(f"ROC AUC:   {np.mean(cv_roc_auc):.4f} ± {np.std(cv_roc_auc):.4f}")
        print(f"Avg Prec:  {np.mean(cv_avg_prec):.4f} ± {np.std(cv_avg_prec):.4f}\n")
        
        # SHAP値を保存
        shap_save_path = self.data_dir / "shap_values.npz"
        np.savez(
            shap_save_path,
            shap_values=self.shap_values_sorted,
            y_true=self.y_true_sorted,
            y_pred_proba=self.y_pred_proba_sorted,
            feature_names=self.X.columns.to_numpy(),
            template_types=self.template_types
        )
        print(f"SHAP値を保存: {shap_save_path}\n")
    
    def analyze_roc_pr_curves(self):
        """ROC曲線とPR曲線を分析"""
        print("=== ROC/PR曲線の分析 ===")
        
        # ROC曲線
        fpr, tpr, thresholds_roc = roc_curve(self.y_true_sorted, self.y_pred_proba_sorted)
        roc_auc = auc(fpr, tpr)
        
        # PR曲線
        precision, recall, thresholds_pr = precision_recall_curve(
            self.y_true_sorted, self.y_pred_proba_sorted
        )
        avg_precision = average_precision_score(self.y_true_sorted, self.y_pred_proba_sorted)
        
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
            y=self.y_true_sorted.mean(), color='navy', linestyle='--', lw=2,
            label=f'Baseline ({self.y_true_sorted.mean():.3f})'
        )
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall', fontsize=12)
        ax2.set_ylabel('Precision', fontsize=12)
        ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax2.legend(loc="lower left")
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "01_model_performance.png", dpi=300, bbox_inches='tight')
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
            y_pred = (self.y_pred_proba_sorted >= threshold).astype(int)
            f1 = f1_score(self.y_true_sorted, y_pred, zero_division=0)
            f1_scores.append(f1)
        
        optimal_threshold_f1 = thresholds_test[np.argmax(f1_scores)]
        max_f1 = np.max(f1_scores)
        
        self.optimal_threshold = optimal_threshold_f1
        
        y_pred_optimal = (self.y_pred_proba_sorted >= optimal_threshold_f1).astype(int)
        acc_optimal = accuracy_score(self.y_true_sorted, y_pred_optimal)
        prec_optimal = precision_score(self.y_true_sorted, y_pred_optimal, zero_division=0)
        rec_optimal = recall_score(self.y_true_sorted, y_pred_optimal, zero_division=0)
        
        # 混同行列を計算
        cm = confusion_matrix(self.y_true_sorted, y_pred_optimal)
        
        # モデルメトリクスを保存
        self.model_metrics = {
            'optimal_threshold': float(optimal_threshold_f1),
            'accuracy': float(acc_optimal),
            'precision': float(prec_optimal),
            'recall': float(rec_optimal),
            'f1_score': float(max_f1),
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
    
    def visualize_feature_consistency(self):
        """特徴の一貫性と寄与度を可視化"""
        print("=== 特徴の一貫性分析 ===")
        
        feature_contribution_stats = []
        
        for i in range(self.shap_values_sorted.shape[1]):
            feature_shap = self.shap_values_sorted[:, i]
            
            positive_ratio = (feature_shap > 0).sum() / len(feature_shap)
            mean_positive = feature_shap[feature_shap > 0].mean() if (feature_shap > 0).any() else 0
            mean_negative = feature_shap[feature_shap < 0].mean() if (feature_shap < 0).any() else 0
            
            feature_contribution_stats.append({
                'feature': self.X.columns[i],
                'positive_ratio': positive_ratio,
                'mean_positive': mean_positive,
                'mean_negative': mean_negative,
                'consistency': abs(2 * positive_ratio - 1),
                'net_contribution': feature_shap.mean()
            })
        
        stats_df = pd.DataFrame(feature_contribution_stats)
        
        # 可視化
        fig, ax = plt.subplots(figsize=(12, 8))
        
        scatter = ax.scatter(
            stats_df['consistency'],
            stats_df['net_contribution'],
            c=stats_df['positive_ratio'],
            s=np.abs(stats_df['mean_positive']) * 1000,
            alpha=0.6,
            cmap='RdYlBu_r'
        )
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.8, color='red', linestyle='--', alpha=0.5, label='High Consistency Threshold')
        
        # 介入候補をハイライト
        intervention_candidates = stats_df[
            (stats_df['consistency'] > 0.8) & (stats_df['net_contribution'] > 0.01)
        ]
        
        for _, row in intervention_candidates.head(15).iterrows():
            ax.annotate(
                row['feature'].replace('feature_', ''),
                (row['consistency'], row['net_contribution']),
                fontsize=8,
                alpha=0.7
            )
        
        ax.set_xlabel('Consistency (一貫性)', fontsize=12)
        ax.set_ylabel('Net SHAP Contribution (純寄与)', fontsize=12)
        ax.set_title(
            '特徴の一貫性と寄与度の分析\n右上: 一貫して迎合性を促進（介入候補）',
            fontsize=14, fontweight='bold'
        )
        plt.colorbar(scatter, label='Positive Ratio', ax=ax)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "04_consistency_analysis.png",
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        
        # CSVとして保存
        stats_df.to_csv(
            self.data_dir / "feature_consistency_stats.csv",
            index=False
        )
        
        print(f"介入候補特徴: {len(intervention_candidates)}個")
        print(f"一貫性分析を保存しました\n")
        
        return stats_df
    
    def visualize_template_heatmap(self):
        """テンプレートタイプ別のヒートマップを作成"""
        print("=== テンプレートタイプ別分析 ===")
        
        # 上位20特徴を選択
        top_20_features = np.argsort(np.abs(self.shap_values_sorted).mean(axis=0))[::-1][:20]
        
        # テンプレートタイプごとの平均SHAP値
        templates = ["base", "I really like", "I really dislike", "I wrote", "I didn't write"]
        heatmap_data = []
        
        for template in templates:
            template_mask = self.template_types == template
            if template_mask.sum() > 0:
                template_shap = self.shap_values_sorted[template_mask][:, top_20_features]
                mean_shap = template_shap.mean(axis=0)
                heatmap_data.append(mean_shap)
            else:
                heatmap_data.append(np.zeros(len(top_20_features)))
        
        heatmap_df = pd.DataFrame(
            heatmap_data,
            index=templates,
            columns=[self.X.columns[i].replace('feature_', '') for i in top_20_features]
        )
        
        # プロット
        fig, ax = plt.subplots(figsize=(16, 6))
        sns.heatmap(
            heatmap_df.T,
            cmap='RdBu_r',
            center=0,
            annot=True,
            fmt='.3f',
            cbar_kws={'label': 'Mean SHAP Value'},
            ax=ax
        )
        ax.set_xlabel('Template Type', fontsize=12)
        ax.set_ylabel('SAE Feature', fontsize=12)
        ax.set_title(
            'テンプレートタイプ別 特徴寄与度\n赤=迎合性促進、青=抑制',
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "05_template_heatmap.png",
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        
        # CSVとして保存
        heatmap_df.to_csv(
            self.data_dir / "template_analysis.csv"
        )
        
        print(f"テンプレート別ヒートマップを保存しました\n")
    
    def analyze_misclassified_samples(self):
        """誤分類サンプルの分析"""
        print("=== 誤分類サンプルの分析 ===")
        
        y_pred = (self.y_pred_proba_sorted >= self.optimal_threshold).astype(int)
        
        false_positives = (self.y_true_sorted == 0) & (y_pred == 1)
        false_negatives = (self.y_true_sorted == 1) & (y_pred == 0)
        true_positives = (self.y_true_sorted == 1) & (y_pred == 1)
        
        print(f"False Positives: {false_positives.sum()}")
        print(f"False Negatives: {false_negatives.sum()}")
        print(f"True Positives: {true_positives.sum()}")
        
        # False Positiveの主要特徴
        if false_positives.sum() > 0:
            fp_shap_values = self.shap_values_sorted[false_positives]
            fp_mean_shap = fp_shap_values.mean(axis=0)
            fp_top_features = np.argsort(np.abs(fp_mean_shap))[::-1][:10]
            
            print("\nFalse Positiveに強く寄与する特徴（ノイズの可能性）:")
            for idx in fp_top_features:
                feature_name = self.X.columns[idx]
                mean_contribution = fp_mean_shap[idx]
                print(f"  {feature_name}: {mean_contribution:.4f}")
        
        # True Positiveの主要特徴
        if true_positives.sum() > 0:
            tp_shap_values = self.shap_values_sorted[true_positives]
            tp_mean_shap = tp_shap_values.mean(axis=0)
            tp_top_features = np.argsort(tp_mean_shap)[::-1][:10]
            
            print("\nTrue Positiveに強く寄与する特徴（確実な介入候補）:")
            for idx in tp_top_features:
                feature_name = self.X.columns[idx]
                mean_contribution = tp_mean_shap[idx]
                print(f"  {feature_name}: {mean_contribution:.4f}")
        
        print()
    
    def find_intervention_features(self):
        """段階的フィルタリングで介入特徴を特定"""
        print("=== 介入特徴の特定 ===")
        
        y_pred = (self.y_pred_proba_sorted >= self.optimal_threshold).astype(int)
        
        # ステップ1: 量的基準（重要度の高い特徴）
        mean_abs_shap = np.abs(self.shap_values_sorted).mean(axis=0)
        important_features = np.where(mean_abs_shap > np.percentile(mean_abs_shap, 90))[0]
        print(f"ステップ1 - 重要特徴候補: {len(important_features)}個")
        
        # ステップ2: 質的基準（迎合性を促進する方向）
        mean_shap = self.shap_values_sorted.mean(axis=0)
        positive_contributors = important_features[mean_shap[important_features] > 0]
        print(f"ステップ2 - 迎合性促進特徴: {len(positive_contributors)}個")
        
        # ステップ3: 一貫性の確認
        consistent_features = []
        for feat_idx in positive_contributors:
            positive_ratio = (self.shap_values_sorted[:, feat_idx] > 0).sum() / len(self.shap_values_sorted)
            if positive_ratio > 0.7:
                consistent_features.append(feat_idx)
        print(f"ステップ3 - 一貫性のある特徴: {len(consistent_features)}個")
        
        # ステップ4: True Positiveでの検証
        tp_mask = (self.y_true_sorted == 1) & (y_pred == 1)
        tp_shap = self.shap_values_sorted[tp_mask]
        
        validated_features = []
        for feat_idx in consistent_features:
            tp_mean = tp_shap[:, feat_idx].mean()
            if tp_mean > 0.01:
                validated_features.append(feat_idx)
        print(f"ステップ4 - 検証済み介入ターゲット: {len(validated_features)}個")
        
        # ステップ5: クラスター分析で重複除去
        if len(validated_features) > 1:
            validated_shap = self.shap_values_sorted[:, validated_features]
            shap_correlation = np.corrcoef(validated_shap.T)
            
            # 類似度が高すぎる特徴を除外（相関 > 0.9）
            unique_features = []
            used = set()
            
            for i, feat_idx in enumerate(validated_features):
                if i in used:
                    continue
                unique_features.append(feat_idx)
                
                # この特徴と高相関の特徴をマーク
                for j in range(i + 1, len(validated_features)):
                    if abs(shap_correlation[i, j]) > 0.9:
                        used.add(j)
            
            final_features = unique_features
        else:
            final_features = validated_features
        
        print(f"ステップ5 - 最終介入特徴: {len(final_features)}個\n")
        
        # 最終的な介入特徴リストを作成
        self.intervention_features = {
            'feature_ids': [int(self.X.columns[i].replace('feature_', '')) for i in final_features],
            'feature_names': [self.X.columns[i] for i in final_features],
            'mean_shap_values': [float(mean_shap[i]) for i in final_features],
            'consistency_scores': [
                float((self.shap_values_sorted[:, i] > 0).sum() / len(self.shap_values_sorted))
                for i in final_features
            ],
            'importance_scores': [float(mean_abs_shap[i]) for i in final_features]
        }
        
        # ソート（重要度順）
        sorted_indices = np.argsort(self.intervention_features['importance_scores'])[::-1]
        for key in self.intervention_features:
            self.intervention_features[key] = [
                self.intervention_features[key][i] for i in sorted_indices
            ]
        
        # 結果を表示
        print("=== 最終介入特徴リスト ===")
        for i, (feat_id, feat_name, shap_val, consistency, importance) in enumerate(zip(
            self.intervention_features['feature_ids'],
            self.intervention_features['feature_names'],
            self.intervention_features['mean_shap_values'],
            self.intervention_features['consistency_scores'],
            self.intervention_features['importance_scores']
        ), 1):
            print(f"{i}. {feat_name} (ID: {feat_id})")
            print(f"   平均SHAP: {shap_val:.4f}, 一貫性: {consistency:.2%}, 重要度: {importance:.4f}")
        
        print()
        
        return self.intervention_features
    
    def create_shap_plots(self):
        """SHAP標準プロットを作成"""
        print("=== SHAP可視化 ===")
        
        explanation = shap.Explanation(
            values=self.shap_values_sorted,
            base_values=None,
            data=self.X.values,
            feature_names=self.X.columns.tolist()
        )
        
        # Beeswarm plot
        plt.figure(figsize=(12, 10))
        shap.plots.beeswarm(explanation, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "02_shap_beeswarm.png",
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        
        # Bar plot
        plt.figure(figsize=(12, 10))
        shap.plots.bar(explanation, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "03_shap_bar.png",
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        
        print(f"SHAP可視化を保存しました\n")
    
    def save_results(self):
        """分析結果を保存"""
        print("=== 結果の保存 ===")
        
        # 実験設定を保存 (config.json)
        config_file = self.experiment_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump({
                'experiment_name': self.file_prefix,
                'timestamp': self.timestamp,
                'input_file': str(self.input_file),
                'token_position': self.token_position,
                'data_shape': {
                    'n_samples': int(self.X.shape[0]),
                    'n_features': int(self.X.shape[1]),
                    'n_class_0': int((self.y == 0).sum()),
                    'n_class_1': int((self.y == 1).sum())
                },
                'model_config': self.lgb_params,
                'execution_time': datetime.now().isoformat()
            }, f, indent=2)
        print(f"実験設定を保存: {config_file}")
        
        # クロスバリデーション結果を保存
        cv_results_file = self.data_dir / "cv_results.json"
        cv_summary = {
            'n_splits': len(self.cv_metrics),
            'fold_results': self.cv_metrics,
            'summary_statistics': {
                'accuracy_mean': float(np.mean([m['accuracy'] for m in self.cv_metrics])),
                'accuracy_std': float(np.std([m['accuracy'] for m in self.cv_metrics])),
                'precision_mean': float(np.mean([m['precision'] for m in self.cv_metrics])),
                'precision_std': float(np.std([m['precision'] for m in self.cv_metrics])),
                'recall_mean': float(np.mean([m['recall'] for m in self.cv_metrics])),
                'recall_std': float(np.std([m['recall'] for m in self.cv_metrics])),
                'f1_mean': float(np.mean([m['f1'] for m in self.cv_metrics])),
                'f1_std': float(np.std([m['f1'] for m in self.cv_metrics])),
                'roc_auc_mean': float(np.mean([m['roc_auc'] for m in self.cv_metrics])),
                'roc_auc_std': float(np.std([m['roc_auc'] for m in self.cv_metrics])),
                'avg_precision_mean': float(np.mean([m['avg_precision'] for m in self.cv_metrics])),
                'avg_precision_std': float(np.std([m['avg_precision'] for m in self.cv_metrics]))
            }
        }
        with open(cv_results_file, 'w') as f:
            json.dump(cv_summary, f, indent=2)
        print(f"クロスバリデーション結果を保存: {cv_results_file}")
        
        # SHAP値の統計サマリーを計算・保存
        shap_stats_file = self.data_dir / "shap_statistics.csv"
        shap_stats = []
        for i, feature_name in enumerate(self.X.columns):
            feature_shap = self.shap_values_sorted[:, i]
            shap_stats.append({
                'feature_name': feature_name,
                'feature_id': int(feature_name.replace('feature_', '')),
                'mean_abs_shap': float(np.abs(feature_shap).mean()),
                'mean_shap': float(feature_shap.mean()),
                'median_shap': float(np.median(feature_shap)),
                'std_shap': float(feature_shap.std()),
                'max_shap': float(feature_shap.max()),
                'min_shap': float(feature_shap.min()),
                'positive_ratio': float((feature_shap > 0).sum() / len(feature_shap))
            })
        shap_stats_df = pd.DataFrame(shap_stats)
        shap_stats_df = shap_stats_df.sort_values('mean_abs_shap', ascending=False)
        shap_stats_df.to_csv(shap_stats_file, index=False)
        print(f"SHAP統計サマリーを保存: {shap_stats_file}")
        
        # Top-k特徴のランキングを保存
        top_k = min(50, len(self.X.columns))
        top_features_file = self.data_dir / f"top{top_k}_features.csv"
        top_features_df = shap_stats_df.head(top_k)
        top_features_df.to_csv(top_features_file, index=False)
        print(f"Top-{top_k}特徴を保存: {top_features_file}")
        
        # 介入特徴リストをJSON形式で保存
        intervention_file = self.data_dir / "intervention_features.json"
        with open(intervention_file, 'w') as f:
            json.dump({
                'token_position': self.token_position,
                'input_file': self.input_file,
                'timestamp': datetime.now().isoformat(),
                'model_hyperparameters': self.lgb_params,
                'model_performance': {
                    'roc_auc': float(self.roc_auc),
                    'average_precision': float(self.avg_precision),
                    **self.model_metrics
                },
                'cross_validation_summary': cv_summary['summary_statistics'],
                'optimal_threshold': float(self.optimal_threshold),
                'intervention_features': self.intervention_features,
                'summary': {
                    'total_features': self.X.shape[1],
                    'total_samples': self.X.shape[0],
                    'intervention_feature_count': len(self.intervention_features['feature_ids'])
                }
            }, f, indent=2)
        
        print(f"介入特徴リストを保存: {intervention_file}")
        
        # 統計サマリーを保存
        summary_file = self.experiment_dir / "summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"=== SAE介入特徴分析サマリー ===\n\n")
            f.write(f"入力ファイル: {self.input_file}\n")
            f.write(f"トークン位置: {self.token_position}\n")
            f.write(f"分析日時: {datetime.now().isoformat()}\n\n")
            
            f.write(f"=== データ統計 ===\n")
            f.write(f"総特徴数: {self.X.shape[1]}\n")
            f.write(f"総サンプル数: {self.X.shape[0]}\n")
            f.write(f"Flag=0 (非迎合): {(self.y == 0).sum()} ({(self.y == 0).sum() / len(self.y) * 100:.1f}%)\n")
            f.write(f"Flag=1 (迎合): {(self.y == 1).sum()} ({(self.y == 1).sum() / len(self.y) * 100:.1f}%)\n\n")
            
            f.write(f"=== モデルハイパーパラメータ ===\n")
            for key, value in self.lgb_params.items():
                f.write(f"{key}: {value}\n")
            f.write(f"\n")
            
            f.write(f"=== クロスバリデーション結果 ({self.lgb_params['n_splits']}-Fold) ===\n")
            cv_acc = [m['accuracy'] for m in self.cv_metrics]
            cv_prec = [m['precision'] for m in self.cv_metrics]
            cv_rec = [m['recall'] for m in self.cv_metrics]
            cv_f1 = [m['f1'] for m in self.cv_metrics]
            cv_roc = [m['roc_auc'] for m in self.cv_metrics]
            cv_ap = [m['avg_precision'] for m in self.cv_metrics]
            
            f.write(f"Accuracy:        {np.mean(cv_acc):.4f} ± {np.std(cv_acc):.4f}\n")
            f.write(f"Precision:       {np.mean(cv_prec):.4f} ± {np.std(cv_prec):.4f}\n")
            f.write(f"Recall:          {np.mean(cv_rec):.4f} ± {np.std(cv_rec):.4f}\n")
            f.write(f"F1 Score:        {np.mean(cv_f1):.4f} ± {np.std(cv_f1):.4f}\n")
            f.write(f"ROC AUC:         {np.mean(cv_roc):.4f} ± {np.std(cv_roc):.4f}\n")
            f.write(f"Avg Precision:   {np.mean(cv_ap):.4f} ± {np.std(cv_ap):.4f}\n\n")
            
            f.write(f"=== 最終モデル性能（全データ統合） ===\n")
            f.write(f"ROC AUC:         {self.roc_auc:.4f}\n")
            f.write(f"Average Precision: {self.avg_precision:.4f}\n")
            f.write(f"最適閾値:        {self.optimal_threshold:.4f}\n")
            f.write(f"Accuracy:        {self.model_metrics['accuracy']:.4f}\n")
            f.write(f"Precision:       {self.model_metrics['precision']:.4f}\n")
            f.write(f"Recall:          {self.model_metrics['recall']:.4f}\n")
            f.write(f"F1 Score:        {self.model_metrics['f1_score']:.4f}\n\n")
            
            f.write(f"=== 混同行列 ===\n")
            f.write(f"                 予測: 非迎合  予測: 迎合\n")
            f.write(f"実際: 非迎合      {self.model_metrics['true_negatives']:>6}      {self.model_metrics['false_positives']:>6}\n")
            f.write(f"実際: 迎合        {self.model_metrics['false_negatives']:>6}      {self.model_metrics['true_positives']:>6}\n\n")
            
            f.write(f"=== 介入特徴 ===\n")
            f.write(f"特定された介入特徴数: {len(self.intervention_features['feature_ids'])}\n\n")
            f.write(f"介入特徴詳細:\n")
            for i, (feat_id, feat_name, shap_val, consistency, importance) in enumerate(zip(
                self.intervention_features['feature_ids'],
                self.intervention_features['feature_names'],
                self.intervention_features['mean_shap_values'],
                self.intervention_features['consistency_scores'],
                self.intervention_features['importance_scores']
            ), 1):
                f.write(f"  {i}. {feat_name} (ID: {feat_id})\n")
                f.write(f"     平均SHAP: {shap_val:.4f}, 一貫性: {consistency:.2%}, 重要度: {importance:.4f}\n")
        
        print(f"サマリーを保存: {summary_file}\n")
    
    def run_full_analysis(self):
        """完全な分析パイプラインを実行"""
        print("=" * 60)
        print("SAE介入特徴探索プログラム - 完全分析")
        print("=" * 60)
        print()
        
        # データ読み込み
        self.load_data()
        
        # モデル学習とSHAP計算
        self.train_model_and_compute_shap()
        
        # ROC/PR曲線分析
        self.analyze_roc_pr_curves()
        
        # 特徴の一貫性分析
        self.visualize_feature_consistency()
        
        # テンプレート別分析
        self.visualize_template_heatmap()
        
        # 誤分類サンプル分析
        self.analyze_misclassified_samples()
        
        # 介入特徴の特定
        self.find_intervention_features()
        
        # SHAP可視化
        self.create_shap_plots()
        
        # 結果保存
        self.save_results()
        
        print("=" * 60)
        print("分析完了！")
        print(f"実験ディレクトリ: {self.experiment_dir}")
        print(f"  - データ: {self.data_dir}")
        print(f"  - 図: {self.figures_dir}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="SAE特徴の介入ターゲット特定プログラム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python find_intervention_features.py --input combined_feedback_data.json --token_position prompt_last_token
  python find_intervention_features.py --input combined_feedback_data_v2.json --token_position response_first_token --output results_v2
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
    
    args = parser.parse_args()
    
    # 分析実行
    finder = InterventionFeatureFinder(
        input_file=args.input,
        token_position=args.token_position,
        output_dir=args.output
    )
    
    finder.run_full_analysis()


if __name__ == "__main__":
    main()
