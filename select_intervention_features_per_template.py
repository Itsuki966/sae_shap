#!/usr/bin/env python3
"""
Template Type別介入候補特徴量選定スクリプト

各template_typeごとにAtPスコアを計算し、各typeでトップ15を選定後、
それらを統合してユニークな介入候補リストを作成します。

使用方法:
    python select_intervention_features_per_template.py --input atp_results.json --top_k_per_template 15
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

# seabornのスタイル設定
sns.set_style("whitegrid")

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Hiragino Kaku Gothic Pro', 'Yu Gothic', 'Meirio', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10


def load_atp_data_per_template(filepath: str, token_position: str = 'prompt_last_token') -> Dict[str, pd.DataFrame]:
    """
    combined_feedback_data.jsonを読み込み、template_typeごとにDataFrameに変換する
    
    計算方法:
        1. 各template_typeごとに、そのtypeの迎合サンプル数とbaseサンプル数をカウント
        2. template_typeごとに、AtPスコアと活性値を集計
        3. template_typeごとに、Global Mean AtPとLog Ratioを計算
    
    Args:
        filepath: combined_feedback_data.jsonのパス
        token_position: 使用するトークン位置（デフォルト: 'prompt_last_token'）
    
    Returns:
        Dict[str, pd.DataFrame]: template_typeをキーとしたDataFrameの辞書
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # まず、迎合を誘発するtemplate_typeを特定する（base以外）
    all_template_types = set()
    for result in data['results']:
        for variation in result['variations']:
            template_type = variation.get('template_type', '')
            all_template_types.add(template_type)
    
    # baseを除外して、迎合誘発template_typeのみを対象とする
    sycophancy_template_types = [t for t in all_template_types if t not in ['', 'base']]
    
    print(f"Token position: {token_position}")
    print(f"検出された迎合誘発template types: {sycophancy_template_types}")
    print("-" * 60)
    
    # 各template_typeに対してデータを集計
    results_per_template = {}
    
    for template_type in sycophancy_template_types:
        print(f"\n処理中: {template_type}")
        
        # Step 1: このtemplate_typeでのサンプル数をカウント
        total_sycophancy_samples = 0
        total_base_samples = 0
        
        for result in data['results']:
            for variation in result['variations']:
                # sae_activationsの存在確認
                if 'sae_activations' not in variation:
                    continue
                if token_position not in variation['sae_activations']:
                    continue
                
                var_template_type = variation.get('template_type', '')
                
                # Base時: 全template_typeで共通のbaseを使用
                if var_template_type in ['', 'base']:
                    total_base_samples += 1
                
                # 迎合時: このtemplate_typeのサンプルのみカウント
                elif var_template_type == template_type:
                    atp_analysis = variation.get('atp_analysis')
                    if atp_analysis is not None and 'error' not in atp_analysis:
                        total_sycophancy_samples += 1
        
        print(f"  迎合サンプル数 (N_syc): {total_sycophancy_samples}")
        print(f"  Baseサンプル数 (N_base): {total_base_samples}")
        
        if total_sycophancy_samples == 0:
            print(f"  警告: {template_type}の迎合サンプルが0件のためスキップします")
            continue
        
        # Step 2: 各特徴量のAtPスコア総和と活性値総和を計算
        feature_atp_sum = defaultdict(float)
        feature_activation_count_syc = defaultdict(int)
        feature_activation_count_base = defaultdict(int)
        feature_activation_sum_syc = defaultdict(float)
        feature_activation_sum_base = defaultdict(float)
        
        # 全サンプルを走査
        for result in data['results']:
            for variation in result['variations']:
                # sae_activationsの存在確認
                if 'sae_activations' not in variation:
                    continue
                if token_position not in variation['sae_activations']:
                    continue
                
                var_template_type = variation.get('template_type', '')
                activations_dict = variation['sae_activations'][token_position]
                
                # Base時: 全template_typeで共通
                if var_template_type in ['', 'base']:
                    for feature_id_str, activation in activations_dict.items():
                        feature_id = str(feature_id_str)
                        feature_activation_sum_base[feature_id] += activation
                        if activation > 0:
                            feature_activation_count_base[feature_id] += 1
                
                # 迎合時: このtemplate_typeのサンプルのみ
                elif var_template_type == template_type:
                    atp_analysis = variation.get('atp_analysis')
                    if atp_analysis is None or 'error' in atp_analysis:
                        continue
                    
                    # 活性値を取得
                    for feature_id_str, activation in activations_dict.items():
                        feature_id = str(feature_id_str)
                        feature_activation_sum_syc[feature_id] += activation
                        if activation > 0:
                            feature_activation_count_syc[feature_id] += 1
                    
                    # AtPスコアを取得
                    if 'top_features' in atp_analysis:
                        for feature in atp_analysis['top_features']:
                            feature_id = str(feature['id'])
                            atp_score = feature.get('score')
                            if atp_score is not None:
                                feature_atp_sum[feature_id] += atp_score
        
        # Step 3: Global Mean AtPと平均活性値を計算
        results = []
        all_feature_ids = set(feature_atp_sum.keys()) | set(feature_activation_sum_syc.keys()) | set(feature_activation_sum_base.keys())
        
        for feature_id in all_feature_ids:
            # Global Mean AtP
            total_score = feature_atp_sum[feature_id]
            global_mean_atp = total_score / total_sycophancy_samples
            
            # 参考値: 活性化した時のみの平均AtP
            count_active = feature_activation_count_syc[feature_id]
            conditional_mean_atp = total_score / count_active if count_active > 0 else 0.0
            
            # 全サンプルベースでの平均活性値
            mean_activation_syc = feature_activation_sum_syc[feature_id] / total_sycophancy_samples
            mean_activation_base = feature_activation_sum_base[feature_id] / total_base_samples
            
            # Log Ratio計算（epsilon=1e-6）
            epsilon = 1e-6
            log_ratio = np.log2((mean_activation_syc + epsilon) / (mean_activation_base + epsilon))
            
            results.append({
                'feature_index': int(feature_id),
                'template_type': template_type,
                'global_mean_atp': global_mean_atp,
                'conditional_mean_atp': conditional_mean_atp,
                'mean_activation_syc': mean_activation_syc,
                'mean_activation_base': mean_activation_base,
                'log_ratio': log_ratio,
                'num_samples_active_syc': feature_activation_count_syc[feature_id],
                'num_samples_active_base': feature_activation_count_base[feature_id],
                'num_samples_total_syc': total_sycophancy_samples,
                'num_samples_total_base': total_base_samples,
                'activation_rate_syc': feature_activation_count_syc[feature_id] / total_sycophancy_samples,
                'activation_rate_base': feature_activation_count_base[feature_id] / total_base_samples
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('global_mean_atp', ascending=False).reset_index(drop=True)
        results_per_template[template_type] = df
        
        print(f"  特徴量数: {len(df)}")
        print(f"  正のAtPを持つ特徴量: {(df['global_mean_atp'] > 0).sum()}")
    
    return results_per_template


def select_top_k_per_template(
    results_per_template: Dict[str, pd.DataFrame],
    k: int = 15,
    min_atp_impact: float = 0.0,
    min_log_ratio: float = 0.0
) -> Dict[str, List[int]]:
    """
    各template_typeごとにトップK個の特徴量を選定
    
    選定条件:
        1. global_mean_atp > min_atp_impact（デフォルト: 0、つまり正の値のみ）
        2. log_ratio > min_log_ratio（デフォルト: 0）
        3. 上記条件を満たす中から、global_mean_atpが高い順にK個選定
    
    Args:
        results_per_template: template_typeごとのDataFrame辞書
        k: 各template_typeで選定する特徴量数
        min_atp_impact: 最小AtPスコア閾値
        min_log_ratio: 最小Log Ratio閾値
    
    Returns:
        Dict[str, List[int]]: template_typeごとの特徴量IDリスト
    """
    top_k_per_template = {}
    
    print(f"\n各template_typeでトップ{k}を選定...")
    print(f"  条件: global_mean_atp > {min_atp_impact}, log_ratio > {min_log_ratio}")
    print("-" * 60)
    
    for template_type, df in results_per_template.items():
        # フィルタリング
        filtered = df[
            (df['global_mean_atp'] > min_atp_impact) &
            (df['log_ratio'] > min_log_ratio)
        ].copy()
        
        # すでにglobal_mean_atp降順でソート済み
        top_k = filtered.head(k)
        feature_ids = top_k['feature_index'].tolist()
        
        top_k_per_template[template_type] = feature_ids
        
        print(f"{template_type}:")
        print(f"  フィルタ通過: {len(filtered)} 特徴量")
        print(f"  選定数: {len(feature_ids)} 特徴量")
        if len(feature_ids) > 0:
            print(f"  AtP範囲: {top_k['global_mean_atp'].max():.6f} ~ {top_k['global_mean_atp'].min():.6f}")
        print()
    
    return top_k_per_template


def merge_feature_lists(top_k_per_template: Dict[str, List[int]]) -> List[int]:
    """
    各template_typeの特徴量リストを統合してユニークなリストを作成
    
    Args:
        top_k_per_template: template_typeごとの特徴量IDリスト
    
    Returns:
        List[int]: 統合されたユニークな特徴量IDリスト（ソート済み）
    """
    all_features: Set[int] = set()
    
    for template_type, feature_ids in top_k_per_template.items():
        all_features.update(feature_ids)
    
    merged_list = sorted(list(all_features))
    
    print("=" * 60)
    print("統合結果:")
    print(f"  合計選定数（重複あり）: {sum(len(ids) for ids in top_k_per_template.values())}")
    print(f"  ユニークな特徴量数: {len(merged_list)}")
    print("=" * 60)
    
    return merged_list


def save_results(
    top_k_per_template: Dict[str, List[int]],
    merged_list: List[int],
    results_per_template: Dict[str, pd.DataFrame],
    output_dir: Path,
    args=None
):
    """
    選定結果を保存
    
    Args:
        top_k_per_template: template_typeごとの特徴量リスト
        merged_list: 統合された特徴量リスト
        results_per_template: template_typeごとの全データ
        output_dir: 保存先ディレクトリ
        args: コマンドライン引数
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 統合リストをCSVで保存
    merged_csv_path = output_dir / f"merged_intervention_candidates_{timestamp}.csv"
    merged_df = pd.DataFrame({'feature_index': merged_list})
    merged_df.to_csv(merged_csv_path, index=False, encoding='utf-8')
    print(f"\n✓ 統合リストを保存: {merged_csv_path}")
    
    # 2. template_typeごとの詳細データを保存
    for template_type, feature_ids in top_k_per_template.items():
        df = results_per_template[template_type]
        selected_df = df[df['feature_index'].isin(feature_ids)].copy()
        
        template_csv_path = output_dir / f"candidates_{template_type}_{timestamp}.csv"
        selected_df.to_csv(template_csv_path, index=False, encoding='utf-8')
        print(f"✓ {template_type}の詳細データを保存: {template_csv_path}")
    
    # 3. サマリーテキストを保存
    summary_path = output_dir / f"selection_summary_{timestamp}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Template Type別 介入候補特徴量 選定サマリー\n")
        f.write("=" * 60 + "\n\n")
        
        # 実行パラメータ
        if args is not None:
            f.write("--- 実行パラメータ ---\n")
            f.write(f"入力ファイル: {args.input}\n")
            f.write(f"トークン位置: {args.token_position}\n")
            f.write(f"各template_typeでの選定数: {args.top_k_per_template}\n")
            f.write(f"最小AtPスコア: {args.min_atp}\n")
            f.write(f"最小Log Ratio: {args.min_log_ratio}\n\n")
        
        # 統合結果
        f.write("--- 統合結果 ---\n")
        f.write(f"template_type数: {len(top_k_per_template)}\n")
        f.write(f"合計選定数（重複あり）: {sum(len(ids) for ids in top_k_per_template.values())}\n")
        f.write(f"ユニークな特徴量数: {len(merged_list)}\n\n")
        
        # template_typeごとの統計
        f.write("--- template_typeごとの選定状況 ---\n")
        for template_type, feature_ids in top_k_per_template.items():
            df = results_per_template[template_type]
            selected_df = df[df['feature_index'].isin(feature_ids)]
            
            f.write(f"\n{template_type}:\n")
            f.write(f"  選定数: {len(feature_ids)}\n")
            if len(selected_df) > 0:
                f.write(f"  AtP範囲: {selected_df['global_mean_atp'].max():.6f} ~ {selected_df['global_mean_atp'].min():.6f}\n")
                f.write(f"  Log Ratio範囲: {selected_df['log_ratio'].max():.2f} ~ {selected_df['log_ratio'].min():.2f}\n")
                f.write(f"  上位5特徴量:\n")
                for _, row in selected_df.head(5).iterrows():
                    f.write(f"    - Feature {int(row['feature_index'])}: AtP={row['global_mean_atp']:.6f}, LogRatio={row['log_ratio']:.2f}\n")
        
        # 統合リスト（全て記載）
        f.write("\n" + "=" * 60 + "\n")
        f.write("--- 統合された特徴量IDリスト（全て） ---\n")
        f.write(f"{merged_list}\n")
        
        # 重複状況の分析
        f.write("\n--- 重複分析 ---\n")
        feature_counts = defaultdict(int)
        for feature_ids in top_k_per_template.values():
            for fid in feature_ids:
                feature_counts[fid] += 1
        
        # 重複度ごとにカウント
        overlap_stats = defaultdict(int)
        for count in feature_counts.values():
            overlap_stats[count] += 1
        
        f.write(f"重複していない特徴（1つのtemplateのみ）: {overlap_stats.get(1, 0)}\n")
        for i in range(2, len(top_k_per_template) + 1):
            if i in overlap_stats:
                f.write(f"{i}つのtemplateで選ばれた特徴: {overlap_stats[i]}\n")
        
        # 最も多く選ばれた特徴（全template_typeで選ばれたもの）
        max_overlap = max(feature_counts.values())
        if max_overlap == len(top_k_per_template):
            highly_common = [fid for fid, count in feature_counts.items() if count == max_overlap]
            f.write(f"\n全template_typeで選ばれた共通特徴（{len(highly_common)}個）:\n")
            f.write(f"{sorted(highly_common)}\n")
    
    print(f"✓ サマリーを保存: {summary_path}")


def visualize_results(
    results_per_template: Dict[str, pd.DataFrame],
    top_k_per_template: Dict[str, List[int]],
    merged_list: List[int],
    output_dir: Path
):
    """
    template_typeごとの選定結果を可視化
    
    各template_typeごとに散布図を作成し、最後に統合リストの重複状況を可視化
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # template_typeごとの散布図
    n_templates = len(results_per_template)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for idx, (template_type, df) in enumerate(results_per_template.items()):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        feature_ids = top_k_per_template[template_type]
        selected_df = df[df['feature_index'].isin(feature_ids)]
        
        # 全特徴量
        ax.scatter(
            df['log_ratio'],
            df['global_mean_atp'],
            c='gray',
            alpha=0.3,
            s=20,
            label='All Features'
        )
        
        # 選定された特徴量
        ax.scatter(
            selected_df['log_ratio'],
            selected_df['global_mean_atp'],
            c='red',
            alpha=0.8,
            s=50,
            label=f'Selected (n={len(selected_df)})',
            edgecolors='darkred',
            linewidths=1
        )
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axvline(x=0, color='blue', linestyle='--', linewidth=0.8, alpha=0.5)
        
        ax.set_xlabel('Log Ratio (迎合特異性)', fontsize=11)
        ax.set_ylabel('Global Mean AtP (因果効果)', fontsize=11)
        ax.set_title(f'{template_type}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # 使用していないサブプロットを非表示
    for idx in range(len(results_per_template), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    fig_path = output_dir / f"per_template_selection_{timestamp}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ template_type別可視化を保存: {fig_path}")
    
    # 重複状況の可視化（棒グラフ）
    feature_counts = defaultdict(int)
    for feature_ids in top_k_per_template.values():
        for fid in feature_ids:
            feature_counts[fid] += 1
    
    overlap_stats = defaultdict(int)
    for count in feature_counts.values():
        overlap_stats[count] += 1
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    x_labels = [f'{i}つのtemplate' for i in range(1, len(top_k_per_template) + 1)]
    y_values = [overlap_stats.get(i, 0) for i in range(1, len(top_k_per_template) + 1)]
    
    bars = ax2.bar(x_labels, y_values, color='steelblue', edgecolor='black', linewidth=1)
    
    # 棒の上に値を表示
    for bar, val in zip(bars, y_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}',
                ha='center', va='bottom', fontsize=11)
    
    ax2.set_xlabel('重複度', fontsize=12)
    ax2.set_ylabel('特徴量数', fontsize=12)
    ax2.set_title('統合リストにおける特徴量の重複状況', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    overlap_fig_path = output_dir / f"overlap_analysis_{timestamp}.png"
    plt.savefig(overlap_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 重複分析図を保存: {overlap_fig_path}")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="Template Type別AtPスコアに基づく介入候補特徴量の選定"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='atp_calculated_results/atp_results_gemma-2-9b-it_20251201_095948.json',
        help='atp_results.jsonのパス'
    )
    parser.add_argument(
        '--token_position',
        type=str,
        default='prompt_last_token',
        help='使用するトークン位置（デフォルト: prompt_last_token）'
    )
    parser.add_argument(
        '--top_k_per_template',
        type=int,
        default=15,
        help='各template_typeで選定する特徴量数（デフォルト: 15）'
    )
    parser.add_argument(
        '--min_atp',
        type=float,
        default=0.0,
        help='最小AtPスコア閾値（デフォルト: 0.0、正の値のみ）'
    )
    parser.add_argument(
        '--min_log_ratio',
        type=float,
        default=0.0,
        help='最小Log Ratio閾値（デフォルト: 0.0）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/selection_results_per_template',
        help='結果の保存先ディレクトリ'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Template Type別 介入候補特徴量 選定プログラム")
    print("=" * 60)
    print(f"入力ファイル: {args.input}")
    print(f"トークン位置: {args.token_position}")
    print(f"各template_typeでの選定数: {args.top_k_per_template}")
    print(f"最小AtPスコア: {args.min_atp}")
    print(f"最小Log Ratio: {args.min_log_ratio}")
    print("=" * 60)
    
    # Step 1: データ読み込み（template_typeごとに分割）
    print("\n[Step 1] データ読み込みとtemplate_type別集計...")
    results_per_template = load_atp_data_per_template(args.input, token_position=args.token_position)
    
    if len(results_per_template) == 0:
        print("エラー: 処理可能なtemplate_typeが見つかりませんでした")
        return
    
    # Step 2: 各template_typeでトップK選定
    print("\n[Step 2] 各template_typeでトップK選定...")
    top_k_per_template = select_top_k_per_template(
        results_per_template,
        k=args.top_k_per_template,
        min_atp_impact=args.min_atp,
        min_log_ratio=args.min_log_ratio
    )
    
    # Step 3: リストを統合
    print("\n[Step 3] リストの統合...")
    merged_list = merge_feature_lists(top_k_per_template)
    
    # Step 4: 結果保存
    print("\n[Step 4] 結果保存...")
    output_dir = Path(args.output_dir)
    save_results(top_k_per_template, merged_list, results_per_template, output_dir, args)
    
    # Step 5: 可視化
    print("\n[Step 5] 可視化...")
    visualize_results(results_per_template, top_k_per_template, merged_list, output_dir)
    
    print("\n" + "=" * 60)
    print("処理完了！")
    print("=" * 60)
    print(f"\n統合された特徴量数: {len(merged_list)}")
    print(f"特徴量ID（最初の20個）: {merged_list[:20]}")
    if len(merged_list) > 20:
        print("...")


if __name__ == "__main__":
    main()
