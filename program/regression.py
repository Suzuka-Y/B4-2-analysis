import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

def run_regression_analysis(df, output_dir):
    """
    Q1(違和感)とQ2(不気味さ)に対する重回帰分析を実行し，標準化偏回帰係数を比較する．
    """
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    targets = ['q1', 'q2']
    explanatory_cols = ['q3', 'q4', 'q5', 'q6', 'q7']
    
    data_for_reg = df[['PID'] + targets + explanatory_cols].dropna()

    # 被験者内標準化
    def standardize(g):
        return (g - g.mean()) / g.std() if g.std() != 0 else g - g.mean()

    df_std = data_for_reg.groupby('PID')[targets + explanatory_cols].transform(standardize)
    
    # 多重共線性(VIF)の確認
    X = sm.add_constant(df_std[explanatory_cols])
    vif_data = pd.DataFrame({
        "Feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    })
    vif_data.to_csv(os.path.join(output_dir, 'vif_statistics.csv'), index=False)

    # モデル構築と係数の比較
    all_coefs = []
    for target_col in targets:
        model = sm.OLS(df_std[target_col], X).fit()
        
        # 結果の保存
        with open(os.path.join(output_dir, f'regression_summary_{target_col}.txt'), 'w') as f:
            f.write(model.summary().as_text())
        
        temp_df = pd.DataFrame({
            'Target': target_col.upper(),
            'Factor': model.params.index,
            'Beta': model.params.values
        })
        all_coefs.append(temp_df[temp_df['Factor'] != 'const'])

    merged_coefs = pd.concat(all_coefs, ignore_index=True)
    
    # 係数比較プロットの作成
    plt.figure(figsize=(12, 6))
    sns.barplot(data=merged_coefs, x='Factor', y='Beta', hue='Target', palette={'Q1': '#95a5a6', 'Q2': '#e74c3c'})
    plt.title('Comparison of Factors: Strangeness (Q1) vs. Creepiness (Q2)')
    plt.ylabel('Standardized Coefficient')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'comparison_plot.png'))
    plt.close()