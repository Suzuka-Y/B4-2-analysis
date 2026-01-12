import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import os

from program.multico import run_multicollinearity_check

def run_regression(df, output_dir):
    """
    重回帰分析のメイン実行関数
    """

    # 計算パート: 統計処理を行い、結果辞書を取得
    results = calculate_regression(df)
    
    if not results:
        print("[!] Regression analysis failed due to data issues.")
        return

    # 出力パート: 結果を受け取り、ファイル保存とグラフ描画を行う
    save_regression_outputs(results, output_dir)

    # マルチコ検証
    run_multicollinearity_check(df, output_dir)

def calculate_regression(df):
    """
    計算パート
    """
    # Level 2のみ抽出
    if 'Level' in df.columns:
        df_reg = df[df['Level'] == 2].copy()
    else:
        df_reg = df.copy()

    explanatory_vars = ['q3', 'q4', 'q5', 'q6', 'q7']
    targets = ['q1', 'q2']
    
    # 定数項の追加
    X = df_reg[explanatory_vars]
    X = sm.add_constant(X)

    models = {}
    summary_data = []

    for target in targets:
        y = df_reg[target]
        model = sm.OLS(y, X).fit()
        
        models[target] = model
        
        # グラフ用データの蓄積
        for var in explanatory_vars:
            summary_data.append({
                'Target': target.upper(),
                'Factor': var,
                'Coefficient': model.params[var],
                'P_value': model.pvalues[var]
            })
            
    return {
        'n_samples': len(df_reg),
        'explanatory_vars': explanatory_vars,
        'models': models,
        'summary_data': summary_data
    }

def save_regression_outputs(results, output_dir):
    """
    出力パート
    """
    # ディレクトリ準備
    fig_dir = os.path.join(output_dir, 'figures')
    report_path = os.path.join(output_dir, 'regression_report.txt')

    models = results['models']
    explanatory_vars = results['explanatory_vars']
    
    # テキストレポートの作成
    lines = []
    lines.append("Regression Analysis Report")
    lines.append("==========================")
    lines.append(f"Data Points (Level 2 only): {results['n_samples']}")
    lines.append(f"Explanatory Variables: {', '.join(explanatory_vars)}")
    lines.append("-" * 60)

    for target, model in models.items():
        target_label = "Q1 (Strangeness)" if target == 'q1' else "Q2 (Creepiness)"
        lines.append(f"\n[Target Variable: {target_label}]")
        lines.append(f"R-squared: {model.rsquared:.4f}")
        lines.append(f"Adj. R-squared: {model.rsquared_adj:.4f}")
        lines.append(f"F-statistic: {model.fvalue:.4f} (p={model.f_pvalue:.4e})")
        lines.append("\nCoefficients:")
        lines.append(f"{'Factor':<10} {'Coef (Beta)':>12} {'Std.Err':>10} {'t':>8} {'P>|t|':>8} {'Sig':>4}")
        lines.append("-" * 75)
        
        for var in explanatory_vars:
            coef = model.params[var]
            std_err = model.bse[var]
            t_val = model.tvalues[var]
            p_val = model.pvalues[var]
            sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            
            lines.append(f"{var:<10} {coef:12.4f} {std_err:10.4f} {t_val:8.3f} {p_val:8.4f} {sig:>4}")
        
        lines.append("-" * 75)

    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"regression report saved: {report_path}")
    except Exception as e:
        print(f"[!] Failed to save report: {e}")

    # グラフの描画
    if results['summary_data']:
        res_df = pd.DataFrame(results['summary_data'])
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Factor', y='Coefficient', hue='Target', data=res_df, palette='viridis')
        plt.axhline(0, color='black', linewidth=0.8)
        
        plt.title('Comparison of Standardized Coefficients (Beta)\nQ1 vs Q2')
        plt.ylabel('Standardized Beta')
        plt.xlabel('Explanatory Factors')
        plt.legend(title='Target Variable')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        img_path = os.path.join(fig_dir, 'reg_comparison.png')
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()

        print(f"regression comparison plot saved: {img_path}")