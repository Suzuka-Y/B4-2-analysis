import os
from program.format_data import format_data 
from program.demographics import run_demographics
from program.validation import run_validation
from program.check_strength import run_strength_check
from program.regression import run_regression_analysis
from program.clean_for_qualitative import clean_data_for_qualitative
from program.qualitative import run_qualitative_analysis

# 設定: データディレクトリと出力先
RAW_DATA_DIR = '.'
OUTPUT_DIR = './generated'

def main():
    print("=== Analysis Pipeline Started ===")
    
    # 1. データ整形 (Raw Data -> Tidy Data)
    # 実験ログと記述回答を結合し，分析可能な形式へ変換する
    df = format_data(RAW_DATA_DIR, OUTPUT_DIR)
    
    if df is None:
        print("[Error] Data formatting failed.")
        return

    # 2. 参加者属性の集計 (Demographics)
    # 年齢・性別等の基本統計量を算出する
    run_demographics(df, OUTPUT_DIR)

    # 3. 操作チェックと妥当性検証 (Validation & Strength Check)
    # 各実験条件が意図通りに機能したかをt検定により検証する
    run_validation(df)
    
    # 操作強度の均質性を分散分析(ANOVA)により確認する
    run_strength_check(df, OUTPUT_DIR)

    # 4. 重回帰分析 (Regression Analysis)
    # 違和感(Q1)および不気味さ(Q2)の要因を特定する
    run_regression_analysis(df, OUTPUT_DIR)

    # 5. 定性データの匿名化 (Anonymization)
    # テキスト分析に先立ち，個人特定につながる情報を削除する
    tidy_file_path = os.path.join(OUTPUT_DIR, 'integrated_tidy_data.csv')
    df_anon = clean_data_for_qualitative(tidy_file_path, OUTPUT_DIR)
    
    # 6. 定性分析 (Qualitative Analysis)
    # 自由記述回答に対する形態素解析および頻出語分析を行う
    if df_anon is not None:
        anon_file_path = os.path.join(OUTPUT_DIR, 'qualitative_data_anonymized.csv')
        run_qualitative_analysis(anon_file_path, OUTPUT_DIR)

    print("\n=== All Analysis Steps Completed Successfully ===")

if __name__ == "__main__":
    main()