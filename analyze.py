import os
from program.format_data import format_data 
from program.demographics import run_demographics
from program.clean_for_qualitative import clean_data_for_qualitative
from program.validation import run_validation
from program.standardize import run_standardize
from program.check_strength import run_strength_check
from program.regression import run_regression
from program.qualitative import run_qualitative_analysis

# 設定: データディレクトリと出力先
RAW_DATA_DIR = '.'
OUTPUT_DIR = './generated'

def main():
    print("=== Analysis Pipeline Started ===")
    
    # 1. データ整形: 実験ログと記述回答を結合し，分析可能な形式へ変換する
    df = format_data(RAW_DATA_DIR, OUTPUT_DIR)
    # 1段階目の刺激データを扱わないのであればこれらを通す
    df = df[(df['Category'] == 'base') | (df['Level'] == 2)].copy()
    df.to_csv(os.path.join(OUTPUT_DIR, 'integrated_tidy_data.csv'), index=False, encoding='utf-8-sig')

    # 2. 参加者属性の集計: 年齢・性別等の基本統計量を算出する
    run_demographics(df, OUTPUT_DIR)

    # 3. 定性データの匿名化: 個人特定につながる情報（年齢・性別・所要時間）を削除する
    tidy_file_path = os.path.join(OUTPUT_DIR, 'integrated_tidy_data.csv')
    df_anon = clean_data_for_qualitative(tidy_file_path, OUTPUT_DIR)
    # 欠損値（-1）を1に置換
    df_anon[['q7']] = df_anon[['q7']].replace(-1, 1)

    # 4. 操作チェックと妥当性検証: 各実験条件が意図通りに機能したかをt検定により検証する
    run_validation(df_anon, OUTPUT_DIR)

    # 5. 被験者内標準化
    df_std = run_standardize(df_anon, OUTPUT_DIR)
    
    # 6. 操作強度の均質性検証（分散分析）-> 結果が均一でなかった場合はTukey-Kramer法を用いた多重比較を行う
    run_strength_check(df_std, OUTPUT_DIR)

    # 7. 重回帰分析
    run_regression(df_std, OUTPUT_DIR)
    
    # 8. 定性分析: 自由記述回答に対する形態素解析・頻出語分析
    if df_anon is not None:
        run_qualitative_analysis(df_anon, OUTPUT_DIR)

    print("=== All Analysis Steps Completed Successfully ===")

if __name__ == "__main__":
    main()