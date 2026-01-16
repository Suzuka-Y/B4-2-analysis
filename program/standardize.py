import pandas as pd

def run_standardize(df_anon, output_dir):
    # 標準化の対象となるColumn
    target_cols = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7']

    # 標準化関数
    def standardize(g):
        # 注意: pandasのstd()は不偏標準偏差(ddof=1)を計算する
        std = g.std()
        mean = g.mean()
        
        # 標準偏差が0（すべての回答が同じ）または計算不能（データが1つ以下）の場合
        if pd.isna(std) or std == 0:
            return g - mean # 平均を引く（0を返却）
        
        return (g - mean) / std

    # 被験者内標準化: PIDごとにグループ化し、各Columnに対して標準化
    df_std = df_anon.copy()
    df_std[target_cols] = df_std.groupby('PID')[target_cols].transform(standardize)

    # CSV形式で保存
    output_path = output_dir+'/standardized_data.csv'
    df_std.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n[i] Standerdized data saved: {output_path}")
    return df_std