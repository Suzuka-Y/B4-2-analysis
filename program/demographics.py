import os

def run_demographics(df, output_dir):
    """
    参加者の属性（年齢・性別・所要時間）を集計し，レポートを出力する．
    """
    report_path = os.path.join(output_dir, 'demographics_report.txt')
    
    # PIDごとのユニークな属性データを抽出
    col_map = {c.lower(): c for c in df.columns}
    pid_col = col_map.get('pid')
    age_col = col_map.get('age')
    sex_col = col_map.get('sex')
    time_col = col_map.get('exptime')
    
    if not (pid_col and age_col and sex_col and time_col):
        return

    cols = [pid_col, age_col, sex_col, time_col]
    
    participants = df.groupby(pid_col)[cols].first()
    
    with open(report_path, 'w') as f:
        f.write("=== Demographics Report ===\n\n")
        f.write(f"Total Participants (N): {len(participants)}\n\n")
        
        # 年齢の要約統計量
        age_stats = participants[age_col].describe()
        f.write(f"--- Age ---\nMean: {age_stats['mean']:.2f}, SD: {age_stats['std']:.2f}\n")
        f.write(f"Range: {age_stats['min']} - {age_stats['max']}\n\n")
        
        # 性別分布
        f.write(f"--- Gender ---\n")
        counts = participants[sex_col].value_counts()
        ratios = participants[sex_col].value_counts(normalize=True) * 100
        for label in counts.index:
            f.write(f"{label}: {counts[label]} ({ratios[label]:.1f}%)\n")
        
        # 回答時間の要約統計量
        if time_col:
            t_stats = participants[time_col].describe()
            f.write(f"\n--- Response Time ---\nMean: {t_stats['mean']:.2f}, SD: {t_stats['std']:.2f}\n")
    
    print(f"demographics report saved: {report_path}")