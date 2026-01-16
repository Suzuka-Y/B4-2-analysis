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
    各操作変数の妥当性を検証する．
    Base条件と刺激条件（Level 1, Level 2 それぞれ）の間で対応のあるt検定を行う．
    """
    # 出力バッファ
    lines = []
    lines.append("\n" + "="*80)
    lines.append("  Step 2: Manipulation Check (Validity Test) - Level 1 & 2")
    lines.append("="*80)
    lines.append("Comparing Stimulus Score (Level 1/2) vs Base Score (Level 0)")
    lines.append("Test: Paired t-test (One-sided: Stimulus > Base)")
    lines.append("-" * 80)

    results = []

    base_data = df[df['Category'] == 'base'].copy()

    for category, target_q in TARGET_MAP.items():
        # Categoryが一致する行からLevelのユニーク値を抽出してソート
        levels = sorted(df[df['Category'] == category]['Level'].unique())

        if not levels:
            continue

        # Baseの該当変数をPIDインデックス化
        base_series = base_data.set_index('PID')[target_q]

        for level in levels:
            stim_series = df[
                (df['Category'] == category) & 
                (df['Level'] == level)
            ][['PID', target_q]].groupby('PID').mean()[target_q]

            # 内部結合 (BaseとStimの両方に回答があるPIDのみ残す)
            merged = pd.concat([base_series, stim_series], axis=1, join='inner')
            merged.columns = ['Base', 'Stim']
            
            if merged.empty:
                continue

            scores_base = merged['Base']
            scores_stim = merged['Stim']
            
            # 対応のあるt検定 (片側検定: 刺激 > Base)
            try:
                t_stat, p_val = stats.ttest_rel(scores_stim, scores_base, alternative='greater')
                significance = "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
            except Exception:
                t_stat, p_val, significance = np.nan, np.nan, "-"

            results.append({
                'Category': category,
                'Level': level,
                'Target_Q': target_q,
                'Mean_Base': scores_base.mean(),
                'Mean_Stim': scores_stim.mean(),
                'Diff': scores_stim.mean() - scores_base.mean(),
                't_stat': t_stat,
                'p_val': p_val,
                'Significance': significance
            })

    # 結果の表示と保存
    if results:
        # テーブルヘッダー (Level列を追加)
        header = f"{'Category':>10} {'Lvl':>3} {'Target':>6} {'Mean_Base':>9} {'Mean_Stim':>9} {'Diff':>6} {'t-stat':>8} {'p-val':>8} {'Sig':>4}"
        lines.append(header)
        lines.append("-" * 90)

        for res in results:
            row_str = (
                f"{res['Category']:>10} {res['Level']:3d} {res['Target_Q']:>6} "
                f"{res['Mean_Base']:9.2f} {res['Mean_Stim']:9.2f} {res['Diff']:6.2f} "
                f"{res['t_stat']:8.3f} {res['p_val']:8.4f} {res['Significance']:>4}"
            )
            lines.append(row_str)
            
        lines.append("-" * 90)
        lines.append("Sig: ** p < 0.01, * p < 0.05")
    else:
        lines.append("No paired data found for validation.")

    output_text = "\n".join(lines)

    # ファイル保存
    save_path = os.path.join(output_dir, 'manipulation_check.txt')
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"\n[i] Manipulation check report saved: {save_path}")
    except Exception as e:
        print(f"\n[!] Failed to save report: {e}")