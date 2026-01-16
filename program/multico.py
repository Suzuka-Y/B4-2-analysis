import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

def run_multicollinearity_check(df, output_dir):
    """
    VIF（分散拡大係数）を計算し，多重共線性の検証を行う
    """

    # 計算パート
    vif_results = calculate_vif(df)
    
    if not vif_results:
        print("[!] VIF calculation failed.")
        return

    # 出力パート
    save_vif_report(vif_results, output_dir)


def calculate_vif(df):
    """
    計算パート: 説明変数間VIFの算出
    """
    df_vif = df[df['Category'] != 'base'].copy()

    if df_vif.empty:
        return None

    # 説明変数
    explanatory_vars = ['q3', 'q4', 'q5', 'q6', 'q7']
    
    # 定数項の追加
    X = df[explanatory_vars]
    X = sm.add_constant(X)

    # VIFの計算
    vif_data = []
    for i in range(X.shape[1]):
        var_name = X.columns[i]
        if var_name == 'const':
            continue
            
        vif = variance_inflation_factor(X.values, i)
        vif_data.append({
            'Factor': var_name,
            'VIF': vif
        })

    return {
        'n_samples': len(df),
        'vif_data': vif_data
    }


def save_vif_report(results, output_dir):
    """
    出力パート
    """
    report_path = os.path.join(output_dir, 'vif_report.txt')
    
    lines = []
    lines.append("Multicollinearity Check Report (VIF)")
    lines.append("====================================")
    lines.append(f"Data Points: {results['n_samples']}")
    lines.append("-" * 50)
    lines.append(f"{'Factor':<10} {'VIF':>10} {'Result':>10}")
    lines.append("-" * 50)

    # 判定基準
    # VIF < 5: 問題なし (Safe)
    # 5 <= VIF < 10: 注意 (Caution)
    # VIF >= 10: 深刻な問題あり (Danger)

    for item in results['vif_data']:
        vif = item['VIF']
        factor = item['Factor']
        
        if vif < 5:
            check = "Safe"
        elif vif < 10:
            check = "Caution"
        else:
            check = "Danger"
            
        lines.append(f"{factor:<10} {vif:10.4f} {check:>10}")

    lines.append("-" * 50)
    lines.append("  VIF < 5.0  : Safe")
    lines.append("  VIF >= 10.0: Danger")
    lines.append("すべて 'Safe' であればマルチコは発生していない")

    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"\n[i] VIF report saved: {report_path}")
    except Exception as e:
        print(f"[!] Failed to save VIF report: {e}")