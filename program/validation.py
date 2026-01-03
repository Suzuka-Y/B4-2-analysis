import pandas as pd
from scipy import stats
import numpy as np

# 検証対象のマッピング（カテゴリ: ターゲット質問）
TARGET_MAP = {
    'position': 'q3',    # 配置 -> Q3
    'size': 'q4',        # サイズ -> Q4
    'lack': 'q5',        # 欠如 -> Q5
    'repetition': 'q6',  # 過剰 -> Q6
    'human': 'q7'        # 人物 -> Q7
}

def run_validation(df):
    """
    各操作変数の妥当性を検証する．Base条件と刺激条件の間で対応のあるt検定を行う．
    """
    print("\n" + "="*50 + "\n  Step 2: Manipulation Check (Validity Test)\n" + "="*50)
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

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        print(res_df.to_string(index=False))
    return res_df