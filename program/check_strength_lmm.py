import pandas as pd
import statsmodels.formula.api as smf
import os

def run_strength_check(df_anon, output_dir):
    """
    各カテゴリの操作強度（Q1の変化量）の均質性を LMM で検証する。
    標準化データではなく、生のデータ(df_anon)を使用する。
    """
    print("\n--- Running Strength Homogeneity Check (LMM) ---")

    # 1. データ準備（差分 Delta の計算）
    df_delta = calculate_deltas(df_anon)
    if df_delta is None:
        return

    # 2. レベルごとに LMM を実行
    for lvl in sorted(df_delta['Level'].unique()):
        print(f"   Checking Level {lvl} ...")
        
        # 該当レベルのデータのみ抽出
        df_lvl = df_delta[df_delta['Level'] == lvl].copy()
        
        try:
            # LMM 実行
            # モデル: Delta_Q1 ~ C(Category) + (1|PID)
            model = smf.mixedlm("Delta_Q1 ~ C(Category)", df_lvl, groups=df_lvl["PID"])
            result = model.fit()
            
            # 結果の保存
            save_strength_report(result, df_lvl, output_dir, lvl)
            
        except Exception as e:
            print(f"   [!] LMM failed for Level {lvl}: {e}")


def calculate_deltas(df):
    """
    各被験者・各刺激について、基準刺激(base)からの変化量(Delta)を計算する
    Delta = Stimulus_Score - Base_Score
    """
    required_cols = ['PID', 'Category', 'Level', 'q1']
    if not all(c in df.columns for c in required_cols):
        print(f"[!] Missing columns for strength check. Need: {required_cols}")
        return None

    # Baseのスコアを辞書化
    base_df = df[df['Category'] == 'base']
    base_map = base_df.groupby('PID')['q1'].mean().to_dict()

    # 操作刺激（Base以外）を抽出
    stim_df = df[df['Category'] != 'base'].copy()
    
    # 差分計算
    def get_delta(row):
        base_val = base_map.get(row['PID'])
        if pd.isna(base_val):
            return None
        return row['q1'] - base_val

    stim_df['Delta_Q1'] = stim_df.apply(get_delta, axis=1)
    stim_df.dropna(subset=['Delta_Q1'], inplace=True)
    
    return stim_df


def save_strength_report(result, df_lvl, output_dir, lvl):
    """
    検定結果をテキストファイルに保存
    """
    report_path = os.path.join(output_dir, f'strength_check_lmm_level{lvl}.txt')
    
    lines = []
    lines.append(f"Strength Homogeneity Check (LMM) - Level {lvl}")
    lines.append("=" * 60)
    lines.append("Dependent Variable: Delta_Q1 (Raw Score Difference from Base)")
    lines.append("Model: Delta_Q1 ~ C(Category) + (1|PID)")
    lines.append("-" * 60)
    
    # 記述統計
    desc = df_lvl.groupby('Category')['Delta_Q1'].agg(['count', 'mean', 'std'])
    lines.append("\n[Descriptive Statistics of Delta_Q1]")
    lines.append(desc.to_string())
    lines.append("\n" + "-" * 60)

    # LMM結果（要約）
    lines.append("\n[LMM Summary]")
    lines.append(result.summary().as_text())
    
    # Omnibus Test (Wald Test)
    hypotheses = [name for name in result.model.exog_names if "C(Category)" in name]
    
    if hypotheses:
        # 修正: scalar=True を指定し、属性を .statistic に変更
        wald = result.wald_test(hypotheses, scalar=True)
        
        f_val = wald.statistic  # ここが修正箇所（.fvalue -> .statistic）
        p_val = wald.pvalue
        
        lines.append("\n" + "-" * 60)
        lines.append(f"[Omnibus Test for Category Effect] (Wald F-test)")
        lines.append(f"F-value: {f_val:.4f}, p-value: {p_val:.4f}")
        
        if p_val < 0.05:
            lines.append(">> Result: Significant difference found among categories.")
            lines.append("   (The manipulation strength is NOT homogeneous.)")
        else:
            lines.append(">> Result: No significant difference found.")
            lines.append("   (The manipulation strength can be considered homogeneous.)")
    else:
        lines.append("\n[!] Could not run Omnibus Test.")

    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"   [i] Report saved: {report_path}")
    except Exception as e:
        print(f"   [!] Failed to save report: {e}")