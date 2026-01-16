import os
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def run_tukey_kramer(stim_df, output_dir, lvl):
    """
    Tukey-Kramer法による多重比較検定を行い，結果をテキストファイルに出力する．
    """
    
    # テキスト出力用のバッファ
    lines = []
    lines.append("="*60)
    lines.append("  Post-hoc Test Results (Tukey-Kramer Method)")
    lines.append("="*60)
    lines.append("-" * 70)

    try:
        # Tukey-Kramer検定の実行
        # endog: 従属変数（変化量）, groups: グループ（カテゴリ）
        tukey = pairwise_tukeyhsd(
            endog=stim_df['Delta_Q1_std'], 
            groups=stim_df['Category'], 
            alpha=0.05
        )

        # 結果をデータフレーム化して整形
        tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
        
        # 読みやすい形式でバッファに追加
        lines.append(f"{'Group 1':<12} | {'Group 2':<12} | {'Mean Diff':>10} | {'p-adj':>8} | {'Significant'}")
        lines.append("-" * 70)
        
        for _, row in tukey_df.iterrows():
            # 有意差がある行にはマークをつける
            sig_mark = "*" if row['reject'] else ""
            res_bool = "Yes" if row['reject'] else "No"
            
            lines.append(
                f"{str(row['group1']):<12} | {str(row['group2']):<12} | "
                f"{row['meandiff']:10.4f} | {row['p-adj']:8.4f} | {res_bool:<5} {sig_mark}"
            )
        
        lines.append("-" * 70)
        lines.append("* : p < 0.05")
        lines.append("- 'Yes' のペアは，操作強度に統計的に明確な差がある")
        lines.append("- Mean Diffが正の値ならGroup 2の方が強く，負ならGroup 1の方が強い")

    except Exception as e:
        lines.append(f"\n[!] Tukey-Kramer test failed: {e}")

    # ファイル保存
    output_path = os.path.join(output_dir, f'post-hoc_level{lvl}.txt')
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f'\n[i] post-hoc test report saved: {output_path}')
    except Exception as e:
        pass