#!/usr/bin/env python3
"""
介入候補特徴量選定スクリプト

SAE特徴量のAtPスコアと活性値データから、迎合性抑制のための
介入候補を選定します。

目的:
    - 迎合時に特異的に働く特徴量を特定
    - 言語能力への副作用を最小化
    - 因果効果（AtP）が高い順にランキング

使用方法:
    python select_intervention_features.py --input atp_results.json --top_k 50
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 日本語フォント設定（macOS）
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")


def load_atp_data(filepath: str) -> pd.DataFrame:
    """
    AtP計算結果のJSONファイルを読み込み、DataFrameに変換する
    
    Notebook実装に合わせた計算方法:
        1. 全迎合サンプル数（エラーなし）をカウント
        2. 各特徴量のAtPスコア総和を計算
        3. Global Mean AtP = 総和 / 全迎合サンプル数
           （活性化しなかった場合は寄与0として扱う）
    
    JSONの構造:
        - results: 各質問のデータ
          - variations: 各テンプレートバリエーション
            - template_type: "base" または迎合誘発テンプレート
            - sycophancy_flag: 0 (base) または 1 (迎合)
            - atp_analysis: AtPスコアと特徴量情報
              - top_features: [{id, score, activation, gradient}, ...]
    
    Args:
        filepath: AtP結果のJSONファイルパス
    
    Returns:
        pd.DataFrame: 特徴量ごとの統計情報を含むDataFrame
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Step 1: 全迎合サンプル数と全baseサンプル数をカウント（エラーがないもののみ）
    total_sycophancy_samples = 0
    total_base_samples = 0
    
    for result in data['results']:
        for variation in result['variations']:
            atp_analysis = variation.get('atp_analysis')
            if atp_analysis is None or 'error' in atp_analysis:
                continue
            
            # baseテンプレート
            if variation.get('template_type') == '' or variation.get('sycophancy_flag') == 0:
                total_base_samples += 1
            # 迎合テンプレート
            else:
                total_sycophancy_samples += 1
    
    print(f"Total sycophancy samples (N_syc): {total_sycophancy_samples}")
    print(f"Total base samples (N_base): {total_base_samples}")
    
    # Step 2: 各特徴量のAtPスコア総和と活性値総和を計算（活性化しなかった場合は0として扱う）
    feature_atp_sum = {}
    feature_activation_count_syc = {}  # 参考用：迎合時に実際に活性化した回数
    feature_activation_count_base = {}  # 参考用：base時に実際に活性化した回数
    feature_activation_sum_syc = {}  # 迎合時の活性値総和（全サンプル）
    feature_activation_sum_base = {}  # base時の活性値総和（全サンプル）
    
    # 全サンプルを走査
    for result in data['results']:
        for variation in result['variations']:
            atp_analysis = variation.get('atp_analysis')
            if atp_analysis is None or 'error' in atp_analysis:
                continue
            
            template_type = variation.get('template_type', '')
            is_base = (template_type == '' or variation.get('sycophancy_flag') == 0)
            
            if 'top_features' in atp_analysis:
                for feature in atp_analysis['top_features']:
                    feature_id = str(feature['id'])
                    activation = feature.get('activation', 0.0)
                    
                    # 初期化
                    if feature_id not in feature_atp_sum:
                        feature_atp_sum[feature_id] = 0.0
                        feature_activation_count_syc[feature_id] = 0
                        feature_activation_count_base[feature_id] = 0
                        feature_activation_sum_syc[feature_id] = 0.0
                        feature_activation_sum_base[feature_id] = 0.0
                    
                    # Baseサンプル
                    if is_base:
                        feature_activation_sum_base[feature_id] += activation
                        if activation > 0:
                            feature_activation_count_base[feature_id] += 1
                    
                    # 迎合サンプル
                    else:
                        atp_score = feature.get('score')
                        if atp_score is not None:
                            feature_atp_sum[feature_id] += atp_score
                        
                        feature_activation_sum_syc[feature_id] += activation
                        if activation > 0:
                            feature_activation_count_syc[feature_id] += 1
    
    # Step 3: Global Mean Attribution スコアと平均活性値を計算
    results = []
    for feature_id, total_score in feature_atp_sum.items():
        # Global Mean AtP: 全迎合サンプル数で割る（非活性時は0として扱う）
        global_mean_atp = total_score / total_sycophancy_samples
        
        # 参考値: 活性化した時のみの平均AtP（Conditional Mean）
        count_active = feature_activation_count_syc[feature_id]
        conditional_mean_atp = total_score / count_active if count_active > 0 else 0.0
        
        # 全サンプルベースでの平均活性値（活性化しなかったサンプルは0として扱う）
        mean_activation_syc = feature_activation_sum_syc[feature_id] / total_sycophancy_samples
        mean_activation_base = feature_activation_sum_base[feature_id] / total_base_samples
        
        results.append({
            'feature_index': int(feature_id),
            'global_mean_atp': global_mean_atp,
            'conditional_mean_atp': conditional_mean_atp,
            'mean_activation_syc': mean_activation_syc,
            'mean_activation_base': mean_activation_base,
            'num_samples_active_syc': feature_activation_count_syc[feature_id],
            'num_samples_active_base': feature_activation_count_base[feature_id],
            'num_samples_total_syc': total_sycophancy_samples,
            'num_samples_total_base': total_base_samples,
            'activation_rate_syc': feature_activation_count_syc[feature_id] / total_sycophancy_samples,
            'activation_rate_base': feature_activation_count_base[feature_id] / total_base_samples
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('feature_index').reset_index(drop=True)
    
    return df


def calculate_log_ratio(df: pd.DataFrame, epsilon: float = 1e-6) -> pd.DataFrame:
    """
    迎合特異性を示すLog Ratioを計算する
    
    Log Ratio = log10((mean_activation_syc + ε) / (mean_activation_base + ε))
    
    ここで、mean_activation_syc と mean_activation_base は
    全サンプルベースでの平均活性値（活性化しなかったサンプルは0として扱う）。
    
    正の値: 迎合時に特異的に活性化
    0付近: base時と同程度
    負の値: base時により活性化（迎合時は抑制）
    
    Args:
        df: 特徴量統計データ
        epsilon: ゼロ除算防止用の微小値
    
    Returns:
        pd.DataFrame: log_ratio列が追加されたDataFrame
    """
    df = df.copy()
    df['log_ratio'] = np.log10(
        (df['mean_activation_syc'] + epsilon) / (df['mean_activation_base'] + epsilon)
    )
    return df


def filter_candidates(
    df: pd.DataFrame,
    min_atp_impact: float = 1e-4,
    min_log_ratio: float = 0.5
) -> pd.DataFrame:
    """
    介入候補の複合フィルタリング
    
    条件（すべてAND）:
        1. global_mean_atp > 0 (正の因果効果のみ)
        2. log_ratio > min_log_ratio (迎合特異性)
        3. global_mean_atp > min_atp_impact (最小影響力)
    
    Args:
        df: 特徴量データ
        min_atp_impact: 最小AtPスコア閾値
        min_log_ratio: 最小Log Ratio閾値
    
    Returns:
        pd.DataFrame: フィルタリング後のDataFrame
    """
    candidates = df[
        (df['global_mean_atp'] > 0) &
        (df['log_ratio'] > min_log_ratio) &
        (df['global_mean_atp'] > min_atp_impact)
    ].copy()
    
    # AtPスコアの降順でソート
    candidates = candidates.sort_values('global_mean_atp', ascending=False)
    
    return candidates


def select_top_k_features(df: pd.DataFrame, k: int = 50) -> List[int]:
    """
    上位K個の特徴量IDを抽出
    
    Args:
        df: フィルタリング済みDataFrame
        k: 選定する特徴量数
    
    Returns:
        List[int]: 特徴量IDのリスト
    """
    top_k = df.head(k)
    return top_k['feature_index'].tolist()


def save_results(df: pd.DataFrame, output_dir: Path, top_k_ids: List[int], args=None):
    """
    選定結果をCSVファイルに保存
    
    Args:
        df: 全候補データ
        top_k_ids: 選定された特徴量IDリスト
        output_dir: 保存先ディレクトリ
        args: コマンドライン引数（入力ファイル名や閾値などを記録）
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 選定された特徴量のデータを保存
    selected_df = df[df['feature_index'].isin(top_k_ids)].copy()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"intervention_candidates_{timestamp}.csv"
    
    selected_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"✓ 選定結果を保存: {csv_path}")
    print(f"  選定数: {len(selected_df)} 特徴量")
    
    # サマリー統計も保存
    summary_path = output_dir / f"selection_summary_{timestamp}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=== 介入候補特徴量 選定サマリー ===\n\n")
        
        # 実行パラメータを記録
        if args is not None:
            f.write("--- 実行パラメータ ---\n")
            f.write(f"入力ファイル: {args.input}\n")
            f.write(f"選定数: Top-{args.top_k}\n")
            f.write(f"最小AtPスコア: {args.min_atp}\n")
            f.write(f"最小Log Ratio: {args.min_log_ratio}\n\n")
        
        f.write(f"選定数: {len(selected_df)} 特徴量\n\n")
        f.write("--- 統計情報 ---\n")
        f.write(selected_df[['global_mean_atp', 'log_ratio', 'mean_activation_syc', 'mean_activation_base']].describe().to_string())
        f.write("\n\n--- 上位10特徴量 ---\n")
        f.write(selected_df.head(10).to_string())
    
    print(f"✓ サマリーを保存: {summary_path}")


def visualize_selection(
    df_all: pd.DataFrame,
    df_selected: pd.DataFrame,
    output_dir: Path
):
    """
    選定された特徴量を散布図で可視化
    
    Args:
        df_all: 全特徴量データ
        df_selected: 選定された特徴量データ
        output_dir: 保存先ディレクトリ
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 全特徴量（グレー）
    ax.scatter(
        df_all['log_ratio'],
        df_all['global_mean_atp'],
        c='gray',
        alpha=0.3,
        s=20,
        label='All Features'
    )
    
    # 選定された特徴量（赤）
    ax.scatter(
        df_selected['log_ratio'],
        df_selected['global_mean_atp'],
        c='red',
        alpha=0.8,
        s=50,
        label=f'Selected (n={len(df_selected)})',
        edgecolors='darkred',
        linewidths=1
    )
    
    # 閾値線
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(x=0.5, color='blue', linestyle='--', linewidth=0.8, alpha=0.5, label='Log Ratio Threshold (0.5)')
    
    ax.set_xlabel('Log Ratio (迎合特異性)', fontsize=12)
    ax.set_ylabel('Global Mean AtP (因果効果)', fontsize=12)
    ax.set_title('介入候補特徴量の選定結果', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = output_dir / f"intervention_selection_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 可視化を保存: {fig_path}")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="AtPスコアに基づく介入候補特徴量の選定"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='AtP計算結果のJSONファイルパス'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=50,
        help='選定する上位特徴量数（デフォルト: 50）'
    )
    parser.add_argument(
        '--min_atp',
        type=float,
        default=1e-4,
        help='最小AtPスコア閾値（デフォルト: 1e-4）'
    )
    parser.add_argument(
        '--min_log_ratio',
        type=float,
        default=0.5,
        help='最小Log Ratio閾値（デフォルト: 0.5）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/selection_results',
        help='結果の保存先ディレクトリ'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("介入候補特徴量 選定プログラム")
    print("=" * 60)
    print(f"入力ファイル: {args.input}")
    print(f"選定数: Top-{args.top_k}")
    print(f"最小AtPスコア: {args.min_atp}")
    print(f"最小Log Ratio: {args.min_log_ratio}")
    print("-" * 60)
    
    # Step 1: データ読み込み
    print("\n[Step 1] データ読み込み...")
    df = load_atp_data(args.input)
    print(f"✓ 全特徴量数: {len(df)}")
    
    # Step 2: Log Ratio計算
    print("\n[Step 2] Log Ratio計算...")
    df = calculate_log_ratio(df)
    print(f"✓ 計算完了")
    
    # Step 3: フィルタリング
    print("\n[Step 3] 複合フィルタリング...")
    candidates = filter_candidates(
        df,
        min_atp_impact=args.min_atp,
        min_log_ratio=args.min_log_ratio
    )
    print(f"✓ フィルタ通過: {len(candidates)} 特徴量")
    
    # Step 4: Top-K選定
    print(f"\n[Step 4] Top-{args.top_k} 選定...")
    top_k_ids = select_top_k_features(candidates, k=args.top_k)
    print(f"✓ 選定完了: {len(top_k_ids)} 特徴量")
    
    # Step 5: 結果保存
    print("\n[Step 5] 結果保存...")
    output_dir = Path(args.output_dir)
    save_results(candidates, output_dir, top_k_ids, args)
    
    # Step 6: 可視化
    print("\n[Step 6] 可視化...")
    selected_df = candidates[candidates['feature_index'].isin(top_k_ids)]
    visualize_selection(df, selected_df, output_dir)
    
    print("\n" + "=" * 60)
    print("処理完了！")
    print("=" * 60)
    print(f"\n選定された特徴量ID（Top-{len(top_k_ids)}）:")
    print(top_k_ids[:10], "..." if len(top_k_ids) > 10 else "")


if __name__ == "__main__":
    main()
