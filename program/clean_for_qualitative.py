import pandas as pd
import os

def clean_data_for_qualitative(input_file, output_dir):
    """
    定性分析のために，個人特定につながる属性情報（年齢・性別・時間等）を削除する．
    """
    try:
        df = pd.read_csv(input_file)
    except Exception:
        return None

    sensitive_cols = [
        'Age', 'age', 'Gender', 'gender', 'Sex', 'sex', 
        'Time', 'time', 'expTime', 'Date'
    ]
    cols_to_drop = [c for c in sensitive_cols if c in df.columns]

    df_clean = df.drop(columns=cols_to_drop)
    output_path = os.path.join(output_dir, 'integrated_tidy_data_anon.csv')
    df_clean.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    return df_clean