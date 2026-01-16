import pandas as pd
import re
import os
import glob

# 実験条件のマッピング定義
CATEGORY_MAP = {
    'position': 1, 'size': 2, 'lack': 3, 'repetition': 4, 'human': 5
}

def parse_text_file(file_path):
    """
    記述回答用テキストファイルを解析し，質問ごとの回答と理由を抽出する
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {os.path.basename(file_path)}: {e}")
        return {}

    qual_data = {}
    
    # 正規表現を用いてブロックごとに回答を抽出
    pattern_block = r"Set Index[:\s]+(\d+)(.*?)(?=Set Index[:\s]+\d+|$)"
    matches = re.findall(pattern_block, content, re.DOTALL)
    
    for set_idx_str, text_block in matches:
        set_idx = int(set_idx_str)
        
        # Q1(違和感)およびQ2(不気味さ)の回答・理由を抽出
        match_q1 = re.search(
            r"A\.Q1\s*解答欄[:\uff1a\s]*(.*?)\s*\n\s*理由[:\uff1a\s]*(.*?)(?=\n\s*A\.Q2)", 
            text_block, re.DOTALL
        )
        match_q2 = re.search(
            r"A\.Q2\s*解答欄[:\uff1a\s]*(.*?)\s*\n\s*理由[:\uff1a\s]*(.*?)(?=$|\n-{3,})", 
            text_block, re.DOTALL
        )

        qual_data[set_idx] = {
            'Q1_Answer': match_q1.group(1).strip() if match_q1 else "",
            'Q1_Reason': match_q1.group(2).strip() if match_q1 else "",
            'Q2_Answer': match_q2.group(1).strip() if match_q2 else "",
            'Q2_Reason': match_q2.group(2).strip() if match_q2 else ""
        }
        
    return qual_data

def format_data(base_dir, output_dir):
    """
    指定ディレクトリ内の定量的データ(CSV)と定性的データ(TXT)を統合する
    """
    quant_dir = os.path.join(base_dir, 'quant_data')
    qual_dir = os.path.join(base_dir, 'qual_data')
    output_file = os.path.join(output_dir, 'integrated_tidy_data.csv')

    csv_files = glob.glob(os.path.join(quant_dir, '*.csv'))
    all_data = []

    for csv_file in csv_files:
        try:
            df_raw = pd.read_csv(csv_file)
        except Exception:
            continue
        
        # 不要なカラムの削除
        if 'SetOrder' in df_raw.columns:
            df_raw.drop(columns=['SetOrder'], inplace=True)

        # 参加者属性(PID, 年齢, 性別等)の取得
        attributes = {col: df_raw.iloc[0][col] for col in ['PID', 'age', 'sex', 'expTime'] if col in df_raw.columns}
        
        # PIDの正規化
        if 'PID' not in attributes or pd.isna(attributes['PID']):
            pid_match = re.search(r'^(\d+)_', os.path.basename(csv_file))
            attributes['PID'] = str(int(pid_match.group(1))) if pid_match else "Unknown"
        else:
            attributes['PID'] = str(int(float(attributes['PID'])))

        # 対応するテキストファイルの読み込み
        txt_path = os.path.join(qual_dir, f"PID={attributes['PID']}.txt")
        qual_map = parse_text_file(txt_path) if os.path.exists(txt_path) else {}

        # データの転置と整形 (Tidy Data化)
        stimulus_cols = [c for c in df_raw.columns if c not in ['PID', 'age', 'sex', 'expTime', 'questions']]
        df_t = df_raw[['questions'] + stimulus_cols].set_index('questions').T.reset_index()
        df_t.rename(columns={'index': 'Stimulus_ID'}, inplace=True)
        
        # 属性情報の付与
        for key, val in attributes.items():
            df_t[key] = val

        # 刺激IDからカテゴリとレベルを抽出
        def parse_stimulus(s):
            if s == 'base': return 'base', 1
            match = re.match(r"([a-z]+)(\d+)", s)
            return (match.group(1), int(match.group(2))) if match else (s, None)
        
        df_t[['Category', 'Level']] = df_t['Stimulus_ID'].apply(lambda x: pd.Series(parse_stimulus(x)))

        # 定性データの結合
        for key in ['Q1_Answer', 'Q1_Reason', 'Q2_Answer', 'Q2_Reason']:
            df_t[key] = df_t.apply(lambda row: qual_map.get(CATEGORY_MAP.get(row['Category']), {}).get(key, ""), axis=1)

        # 数値変換処理
        for c in [col for col in df_t.columns if col.startswith('q') and len(col) == 2]:
            df_t[c] = pd.to_numeric(df_t[c], errors='coerce')
        
        all_data.append(df_t)

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        return final_df
    return None