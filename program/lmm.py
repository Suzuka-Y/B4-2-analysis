import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def run_lmm(df, output_dir):
    """
    線形混合効果モデル (Linear Mixed-Effects Model: LMM) による分析
    被験者(PID)をランダム効果として組み込み、データの階層性を考慮する。
    """
    print("\n--- Running Linear Mixed-Effects Model (LMM) ---")

    # 計算パート
    results = calculate_lmm(df)
    
    if not results:
        print("[!] LMM analysis failed due to data issues.")
        return

    # 出力パート
    save_lmm_outputs(results, output_dir)


def calculate_lmm(df):
    """
    LMM計算パート
    """
    # 基準刺激(base)は変動がないため分析から除外
    df_reg = df[df['Category'] != 'base'].copy()

    if df_reg.empty:
        print("[!] No data found for LMM (check Category column).")
        return None
    
    # PIDが数値型の場合、カテゴリとして扱うために文字列等に変換推奨だが、
    # statsmodelsは自動でグループ化してくれるため、念のため欠損除去だけ行う
    if 'PID' not in df_reg.columns:
        print("[!] PID column not found. LMM requires subject ID.")
        return None

    # 説明変数と目的変数
    # q3:変位, q4:サイズ, q5:欠落, q6:反復, q7:社会的存在
    explanatory_vars = ['q3', 'q4', 'q5', 'q6', 'q7']
    targets = ['q1', 'q2'] # q1:違和感, q2:不気味さ

    # LMMの式定義 (Random Intercept Model)
    # Y ~ X1 + ... + X5 + (1|PID)
    # statsmodelsのmixedlmでは groups引数でランダム効果のグループを指定する
    formula_template = "{} ~ q3 + q4 + q5 + q6 + q7"

    models = {}
    summary_data = []

    for target in targets:
        print(f"   Fitting model for target: {target} ...")
        
        formula = formula_template.format(target)
        
        try:
            # 欠損値を含む行を削除（LMMは欠損に弱いため）
            data_for_model = df_reg.dropna(subset=[target, 'PID'] + explanatory_vars)
            
            # モデル構築と適合
            # groups=data_for_model["PID"] により被験者ごとのランダム切片を設定
            model = smf.mixedlm(formula, data_for_model, groups=data_for_model["PID"])
            result = model.fit()
            
            models[target] = result
            
            # グラフ用データの蓄積 (固定効果の係数のみ抽出)
            params = result.params
            bse = result.bse # 標準誤差
            pvalues = result.pvalues

            for var in explanatory_vars:
                if var in params:
                    summary_data.append({
                        'Target': target,
                        'Factor': var,
                        'Coefficient': params[var],
                        'StdErr': bse[var],
                        'P-Value': pvalues[var]
                    })

        except Exception as e:
            print(f"[!] Error fitting model for {target}: {e}")

    return {
        'models': models,
        'summary_data': summary_data
    }


def save_lmm_outputs(results, output_dir):
    """
    結果の保存とグラフ描画
    """
    models = results['models']
    
    # 1. 分析結果のテキストレポート保存
    report_path = os.path.join(output_dir, 'lmm_report.txt')
    lines = []
    lines.append("Linear Mixed-Effects Model Analysis Report")
    lines.append("========================================\n")
    
    for target, result in models.items():
        lines.append(f"Target Variable: {target}")
        lines.append("-" * 60)
        # summary()の結果を文字列として取得
        lines.append(result.summary().as_text())
        lines.append("\n" + "=" * 60 + "\n")

    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"[i] LMM report saved: {report_path}")
    except Exception as e:
        print(f"[!] Failed to save report: {e}")

    # 2. 係数のグラフ描画
    if results['summary_data']:
        res_df = pd.DataFrame(results['summary_data'])
        
        # 描画設定
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        # 係数の棒グラフ
        ax = sns.barplot(
            x='Factor', 
            y='Coefficient', 
            hue='Target', 
            data=res_df, 
            palette='viridis',
            edgecolor='black'
        )
        
        # エラーバーの追加 (標準誤差)
        # Seabornのbarplotはデフォルトで信頼区間を出すが、ここではモデルから得られた標準誤差を使いたい場合
        # 簡易的にmatplotlibで重ねるか、上記barplotのci=Noneにして自分で描画する等の調整が必要。
        # 今回は傾向を見るためデフォルト(または係数値そのもの)を表示。
        
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.title('Fixed Effects Coefficients (LMM)\nComparison of Factors on Strangeness(Q1) & Creepiness(Q2)')
        plt.ylabel('Coefficient Estimate (Fixed Effect)')
        plt.xlabel('Factors (Explanatory Variables)')
        plt.legend(title='Target Variable')
        
        # グラフ保存
        graph_path = os.path.join(output_dir, 'lmm_coefficients.png')
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[i] Coefficient graph saved: {graph_path}")