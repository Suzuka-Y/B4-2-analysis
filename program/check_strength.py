import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os

from program.post_hoc import run_tukey_kramer

def run_strength_check(df_std, output_dir):
    """
    各カテゴリの操作強度（Q1の変化量）の均質性を確認する．
    引数 df_std は既に被験者内標準化（Z得点化）されていることを前提とする．
    """
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # テキスト出力用バッファ
    lines = []
    lines.append("="*60)
    lines.append("  Manipulation Strength Check Report")
    lines.append("  (Delta Q1 = Stimulus_Z - Base_Z)")
    lines.append("="*60)

    # 必要な列をコピー
    data = df_std[['PID', 'Category', 'q1']].copy()

    # Baseとの差分
    base_map = data[data['Category'] == 'base'].set_index('PID')['q1'].to_dict()
    
    stim_df = data[data['Category'] != 'base'].copy()
    
    # 差分計算
    stim_df['Delta_Q1_std'] = stim_df.apply(
        lambda row: row['q1'] - base_map.get(row['PID']) if row['PID'] in base_map else None, axis=1
    )
    stim_df.dropna(subset=['Delta_Q1_std'], inplace=True)

    # 基本統計量の算出: カテゴリごとに 平均、標準偏差、最小、最大 を計算
    desc_stats = stim_df.groupby('Category')['Delta_Q1_std'].agg(['count', 'mean', 'std', 'min', 'max'])
    
    lines.append("\n[1. Descriptive Statistics per Category]")
    lines.append("各カテゴリの操作強度（変化量の平均）")
    lines.append("-" * 75)
    lines.append(f"{'Category':<12} {'N':>5} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    lines.append("-" * 75)
    
    for cat, row in desc_stats.iterrows():
        lines.append(f"{cat:<12} {int(row['count']):5d} {row['mean']:8.3f} {row['std']:8.3f} {row['min']:8.3f} {row['max']:8.3f}")
    
    lines.append("-" * 75)
    lines.append("Meanが大きいほど，操作による違和感の上昇幅が大きい")

    # 可視化 (箱ひげ図)
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x='Category', 
        y='Delta_Q1_std', 
        hue='Category',
        data=stim_df, 
        palette='Set3', 
        legend=False
    )
    plt.title('Manipulation Strength Check (Standardized Change in Q1)')
    plt.ylabel('Delta Z-Score (Stimulus - Base)')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'strength_check.png'))
    plt.close()

    # 3. ANOVA (一元配置分散分析)
    groups = [group['Delta_Q1_std'].values for name, group in stim_df.groupby('Category')]
    f_stat, p_val = stats.f_oneway(*groups)
    
    lines.append("\n\n[2. ANOVA Results]")
    lines.append("カテゴリ間で操作強度に有意な差があるか（均質かどうか）の検定")
    lines.append("-" * 60)
    lines.append(f"F-statistic: {f_stat:.4f}")
    lines.append(f"p-value:     {p_val:.4f}")
    
    if p_val < 0.05:
        lines.append(">> Result: Significant difference found. (操作強度に偏りあり)")
        run_tukey_kramer(stim_df, output_dir)
    else:
        lines.append(">> Result: No significant difference. (操作強度は概ね均質)")

    # ファイル保存
    output_path = os.path.join(output_dir, 'strength_check.txt')
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"strength check report saved: {output_path}")
    except Exception as e:
        print(f"[!] Failed to save strength check report: {e}")