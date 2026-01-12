import pandas as pd
from scipy import stats
import numpy as np
import os

# 検証対象のマッピング（カテゴリ: ターゲット質問）
TARGET_MAP = {
    'position': 'q3',    # 変位 -> Q3
    'size': 'q4',        # サイズ -> Q4
    'lack': 'q5',        # 欠落 -> Q5
    'repetition': 'q6',  # 反復 -> Q6
    'human': 'q7'        # 社会的存在 -> Q7
}

def run_validation(df, output_dir):
    """
    各操作変数の妥当性を検証する．Base条件と刺激条件の間で対応のあるt検定を行う．
    """
    lines = []
    lines.append("\n" + "="*80)
    lines.append("  Manipulation Check Test")
    lines.append("="*80)

    results = []

    for category, target_q in TARGET_MAP.items():
        # Base条件と刺激条件のスコアをPIDで結合
        base_df = df[df['Category'] == 'base'][['PID', target_q]].set_index('PID')
        stim_df = df[df['Category'] == category][['PID', target_q]].groupby('PID').mean()
        merged = base_df.join(stim_df, lsuffix='_base', rsuffix='_stim', how='inner')
        
        if merged.empty: continue

        scores_base = merged[f'{target_q}_base']
        scores_stim = merged[f'{target_q}_stim']
        
        # 対応のあるt検定 (片側検定: 刺激 > Base)
        try:
            t_stat, p_val = stats.ttest_rel(scores_stim, scores_base, alternative='greater')
            significance = "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        except Exception:
            t_stat, p_val, significance = np.nan, np.nan, "-"

        results.append({
            'Category': category,
            'Target_Q': target_q,
            'Mean_Base': round(scores_base.mean(), 2),
            'Mean_Stim': round(scores_stim.mean(), 2),
            'Diff': round(scores_stim.mean() - scores_base.mean(), 2),
            't-stat': round(t_stat, 3),
            'p-val': round(p_val, 4),
            'Sig': significance
        })

    if results:
        header = f"{'Category':>10} {'Target_Q':>10} {'Mean_Base':>10} {'Mean_Stim':>10} {'Diff':>6} {'t-stat':>8} {'p-val':>8} {'Sig':>4}"
        lines.append(header)
        lines.append("-" * 80)

        for res in results:
            row_str = f"{res['Category']:>10} {res['Target_Q']:>10} {res['Mean_Base']:10.1f} {res['Mean_Stim']:10.1f} {res['Diff']:6.1f} {res['t-stat']:8.3f} {res['p-val']:8.4f} {res['Sig']:>4}"
            lines.append(row_str)
    else:
        lines.append("No paired data found for validation.")
    
    output_text = "\n".join(lines)
    save_path = os.path.join(output_dir, 'manipulation_check.txt')
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"manipulation check report saved: {save_path}")
    except Exception as e:
        print(f"[!] Failed to save report: {e}")