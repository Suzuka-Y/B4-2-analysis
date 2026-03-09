import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_lmm_split_level(df, output_dir):
    """
    Level 2 (弱刺激) と Level 3 (強刺激) を分割して LMM を実行し、
    刺激強度による要因の影響力の違いを比較する。
    """
    print("\n--- Running LMM Split by Level (Level 2 vs Level 3) ---")

    # 基準刺激(base)を除外
    df_reg = df[df['Category'] != 'base'].copy()
    
    # ターゲットとするレベル
    levels = [2, 3]
    
    # 結果格納用
    all_summaries = []

    for lvl in levels:
        print(f"   Analyzing Level {lvl} ...")
        
        # 該当レベルのデータのみ抽出
        df_lvl = df_reg[df_reg['Level'] == lvl].copy()
        
        if df_lvl.empty:
            print(f"   [!] No data for Level {lvl}")
            continue

        # LMM実行
        results = calculate_lmm_for_subset(df_lvl, lvl)
        
        if results:
            # レポート保存
            save_level_report(results, output_dir, lvl)
            # グラフ用データを蓄積
            all_summaries.extend(results['summary_data'])

    # 比較グラフの作成
    if all_summaries:
        plot_level_comparison(all_summaries, output_dir)


def calculate_lmm_for_subset(df, lvl):
    """
    特定のサブセット（レベル）に対してLMMを実行
    """
    explanatory_vars = ['q3', 'q4', 'q5', 'q6', 'q7']
    targets = ['q1', 'q2'] # q1:違和感, q2:不気味さ
    
    # 式は本分析と同じ
    formula_template = "{} ~ q3 + q4 + q5 + q6 + q7"

    models = {}
    summary_data = []

    for target in targets:
        formula = formula_template.format(target)
        try:
            # 欠損除去
            data_for_model = df.dropna(subset=[target, 'PID'] + explanatory_vars)
            
            # LMM Fit
            model = smf.mixedlm(formula, data_for_model, groups=data_for_model["PID"])
            result = model.fit()
            
            models[target] = result
            
            # 係数抽出
            params = result.params
            bse = result.bse
            pvalues = result.pvalues

            for var in explanatory_vars:
                if var in params:
                    summary_data.append({
                        'Level': f"Level {lvl}",
                        'Target': target,
                        'Factor': var,
                        'Coefficient': params[var],
                        'StdErr': bse[var],
                        'P-Value': pvalues[var],
                        'Significant': pvalues[var] < 0.05
                    })

        except Exception as e:
            print(f"   [!] Error fitting {target} at Level {lvl}: {e}")
            return None

    return {'models': models, 'summary_data': summary_data}


def save_level_report(results, output_dir, lvl):
    """
    レベルごとの分析結果をテキスト保存
    """
    report_path = os.path.join(output_dir, f'lmm_report_level{lvl}.txt')
    lines = []
    lines.append(f"LMM Analysis Report - Level {lvl}")
    lines.append("=" * 60 + "\n")
    
    for target, result in results['models'].items():
        lines.append(f"Target Variable: {target}")
        lines.append("-" * 60)
        lines.append(result.summary().as_text())
        lines.append("\n" + "=" * 60 + "\n")

    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"   [i] Report saved: {report_path}")
    except Exception as e:
        print(f"   [!] Failed to save report: {e}")


def plot_level_comparison(summary_data, output_dir):
    """
    Level 2 と Level 3 の係数を比較するグラフを描画
    """
    df_res = pd.DataFrame(summary_data)
    
    # ターゲットごとにグラフを分ける
    for target in ['q1', 'q2']:
        df_target = df_res[df_res['Target'] == target]
        
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        # 棒グラフ: x=Factor, y=Coef, hue=Level
        ax = sns.barplot(
            x='Factor', 
            y='Coefficient', 
            hue='Level', 
            data=df_target, 
            palette='magma', 
            edgecolor='black'
        )
        
        # 0ライン
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        
        plt.title(f'Comparison of Coefficients by Stimulus Intensity\nTarget: {target.upper()}')
        plt.ylabel('Coefficient (Fixed Effect)')
        plt.xlabel('Factors')
        plt.legend(title='Intensity Level')
        
        # 保存
        out_path = os.path.join(output_dir, f'lmm_comparison_level_{target}.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   [i] Comparison graph saved: {out_path}")