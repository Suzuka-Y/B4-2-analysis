import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os

from program.post_hoc import run_tukey_kramer

def run_strength_check(df_std, output_dir):
    """
    各カテゴリの操作強度（Q1の変化量）の均質性を確認する．
    """

    # 1. 計算パート
    results = calculate_strength_stats(df_std)
    
    if not results:
        print("[!] Strength check calculation failed.")
        return

    # 2. 出力パート
    save_strength_outputs(results, output_dir)


def calculate_strength_stats(df_std):
    """
    計算パート
    データの差分計算、記述統計、ANOVA検定を行う。
    """
    if 'Level' not in df_std.columns:
        print("[!] 'Level' column is missing.")
        return None

    # 必要な列だけコピー
    data = df_std[['PID', 'Category', 'Level', 'q1']].copy()

    # Baseとの差分（変化量）の算出
    base_map = data[data['Category'] == 'base'].set_index('PID')['q1'].to_dict()
    
    # 刺激データのみ抽出 (Base以外)
    stim_df_all = data[data['Category'] != 'base'].copy()
    
    # 差分（Stimulus - Base）を計算
    stim_df_all['Delta_Q1_std'] = stim_df_all.apply(
        lambda row: row['q1'] - base_map.get(row['PID']) if row['PID'] in base_map else None, axis=1
    )
    stim_df_all.dropna(subset=['Delta_Q1_std'], inplace=True)
    
    # レベルごとの解析結果を格納する
    level_analyses = {}
    levels = sorted(stim_df_all['Level'].unique())

    for lvl in levels:
        stim_df = stim_df_all[stim_df_all['Level'] == lvl].copy()
        if stim_df.empty:
            continue
            
        # 1. 記述統計
        desc_stats = stim_df.groupby('Category')['Delta_Q1_std'].agg(['count', 'mean', 'std'])
        
        # 2. ANOVA
        groups = [g['Delta_Q1_std'].values for n, g in stim_df.groupby('Category')]
        
        if len(groups) >= 2:
            f_stat, p_val = stats.f_oneway(*groups)
            anova_res = {'f': f_stat, 'p': p_val, 'valid': True}
        else:
            anova_res = {'f': None, 'p': None, 'valid': False}
            
        level_analyses[lvl] = {
            'stim_df': stim_df,
            'desc_stats': desc_stats,
            'anova': anova_res
        }

    return {
        'stim_df_all': stim_df_all,
        'level_analyses': level_analyses
    }


def save_strength_outputs(results, output_dir):
    """
    出力パート
    """
    fig_dir = os.path.join(output_dir, 'figure')
    os.makedirs(fig_dir, exist_ok=True)
    
    stim_df_all = results['stim_df_all']
    level_analyses = results['level_analyses']
    
    # --- 1. 統合グラフの作成 (hue='Level') ---
    stim_df_plot = stim_df_all.copy()
    stim_df_plot['Level'] = stim_df_plot['Level'].astype(str)

    plt.figure(figsize=(12, 7))
    sns.boxplot(
        x='Category', 
        y='Delta_Q1_std', 
        hue='Level',
        data=stim_df_plot, 
        palette='viridis'
    )
    plt.title('Manipulation Strength Check: Comparison by Level')
    plt.ylabel('Strength of Manipulation (Delta Z-Score of Q1)')
    plt.xlabel('Manipulation Category')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.legend(title='Stimulus Level', loc='upper right')
    plt.tight_layout()
    
    plot_path = os.path.join(fig_dir, 'strength_check.png')
    plt.savefig(plot_path)
    plt.close()
    
    # 2. テキストレポート作成 & Post-hoc
    lines = []
    lines.append("="*60)
    lines.append("  Manipulation Strength Check Report (By Level)")
    lines.append("  (Delta Q1 = Stimulus_Z - Base_Z)")
    lines.append("="*60)
    lines.append(f"\n[Graph Output] {plot_path}")

    for lvl, analysis in level_analyses.items():
        lines.append(f"\n{'#'*40}")
        lines.append(f"  Level {lvl} Analysis")
        lines.append(f"{'#'*40}")

        # 記述統計
        lines.append("\n[Descriptive Statistics]")
        lines.append(f"{'Category':<12} {'N':>5} {'Mean':>8} {'Std':>8}")
        lines.append("-" * 40)
        for cat, row in analysis['desc_stats'].iterrows():
            lines.append(f"{cat:<12} {int(row['count']):5d} {row['mean']:8.3f} {row['std']:8.3f}")
        lines.append("-" * 40)

        # ANOVA
        anova = analysis['anova']
        if anova['valid']:
            lines.append(f"\n[ANOVA Results] F={anova['f']:.4f}, p={anova['p']:.4f}")
            
            if anova['p'] < 0.05:
                lines.append(">> Result: Significant difference found (Heterogeneous strength).")
                
                # Post-hoc呼び出し
                try:
                    run_tukey_kramer(analysis['stim_df'], output_dir, lvl)
                    lines.append(f"   (See 'post-hoc_level{lvl}.txt' in output dir)")
                except Exception as e:
                    lines.append(f"   [!] Post-hoc failed: {e}")
            else:
                lines.append(">> Result: No significant difference (Homogeneous strength).")
        else:
            lines.append("\n[ANOVA Results] Not enough categories.")

    # レポート保存
    report_path = os.path.join(output_dir, 'strength_check.txt')
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"\n[i] Strength check report saved: {report_path}")
    except Exception as e:
        print(f"[!] Failed to save strength check report: {e}")