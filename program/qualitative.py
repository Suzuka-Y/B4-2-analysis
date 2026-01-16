import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from janome.tokenizer import Tokenizer
import platform

def run_qualitative_analysis(df, output_dir):
    """
    匿名化データを用いて頻出語分析およびクロス集計を行う
    """

    system = platform.system()
    if system == 'Darwin':  # macOS
        plt.rcParams['font.family'] = 'Hiragino Sans'
    elif system == 'Windows':  # Windows
        plt.rcParams['font.family'] = 'MS Gothic'
    else:
        plt.rcParams['font.family'] = 'IPAexGothic'

    try:
        t = Tokenizer()
    except Exception:
        return

    # テキストカラムの特定
    text_col = next((c for c in df.columns if 'Q2_Reason' in c or 'reason' in c), None)
    if not text_col: return

    stop_words = ['こと', 'よう', 'そう', 'もの', 'それ', 'これ', 'ん', 'の', 'ため', '感じ']
    
    def extract_words(text):
        if not isinstance(text, str): return []
        words = []
        for token in t.tokenize(text):
            if token.part_of_speech.split(',')[0] in ['名詞', '形容詞']:
                word = token.base_form
                if word not in stop_words and len(word) > 1:
                    words.append(word)
        return words

    # カテゴリごとの単語集計
    all_words = []
    category_words = {}

    for cat in df['Category'].unique():
        words_in_cat = []
        for text in df[df['Category'] == cat][text_col]:
            words_in_cat.extend(extract_words(text))
        category_words[cat] = words_in_cat
        all_words.extend(words_in_cat)

    if not all_words: return

    # クロス集計とヒートマップの作成
    top_n_words = [w for w, c in Counter(all_words).most_common(30)]
    cross_tab = pd.DataFrame(0, index=df['Category'].unique(), columns=top_n_words)

    for cat, words in category_words.items():
        counts = Counter(words)
        for word in top_n_words:
            cross_tab.loc[cat, word] = counts[word]

    plt.figure(figsize=(12, 8))
    sns.heatmap(cross_tab.astype(int), annot=True, fmt="d", cmap="YlGnBu")
    plt.title('Word Frequency by Category')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'figures', 'qualitative_heatmap.png')
    plt.savefig(output_path)
    plt.close()

    print(f"\n[i] qualiative heatmap saved: {output_path}")