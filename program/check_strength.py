import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os

def run_strength_check(df, output_dir):
    """
    各カテゴリの操作強度（Q1の変化量）の均質性を確認する．
    個人差を排除するため，被験者内標準化（Z得点化）後の値を用いて比較を行う．
    """
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    data = df[['PID', 'Category', 'q1']].copy()

    # 被験者内標準化 (Within-subject Standardization)
    def standardize(g):
        return (g - g.mean()) / g.std() if g.std() != 0 else g - g.mean()

    data['q1_std'] = data.groupby('PID')['q1'].transform(standardize)

    # Baseとの差分（変化量）の算出
    base_map = data[data['Category'] == 'base'].set_index('PID')['q1_std'].to_dict()
    
    stim_df = data[data['Category'] != 'base'].copy()
    stim_df['Delta_Q1_std'] = stim_df.apply(
        lambda row: row['q1_std'] - base_map.get(row['PID']) if row['PID'] in base_map else None, axis=1
    )
    stim_df.dropna(subset=['Delta_Q1_std'], inplace=True)

    # 可視化 (箱ひげ図)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Category', y='Delta_Q1_std', data=stim_df, palette='Set3')
    plt.title('Manipulation Strength Check (Standardized Change in Q1)')
    plt.ylabel('Delta Z-Score (Stimulus - Base)')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'manipulation_strength_check_std.png'))
    plt.close()

    # 一元配置分散分析 (ANOVA)
    groups = [stim_df[stim_df['Category'] == c]['Delta_Q1_std'] for c in stim_df['Category'].unique()]
    f_stat, p_val = stats.f_oneway(*groups)
    
    print(f"\n--- ANOVA for Manipulation Strength ---\nF: {f_stat:.4f}, p: {p_val:.4f}")