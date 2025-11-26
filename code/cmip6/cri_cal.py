import pandas as pd

# 读取 Excel 文件中的每个 sheet
file_path = 'E:/GEO/etccdi/qpm/multi_year_result_separated.xlsx'
output_file_path = 'E:/GEO/etccdi/qpm/multi_year_rank.xlsx'
sheet_names = ['prcptot', 'r95p', 'r99p', 'sdii', 'rx1day']

with pd.ExcelWriter(output_file_path) as writer:
    for sheet_name in sheet_names:
        # 读取当前 sheet
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        # 只保留需要的列
        df = df[['Model', 'Correlation', 'NSE', 'KGE', 'Taylor_Skill_Score','NRMSE']]

        # 排序并替换为排名
        df['NRMSE'] = df['NRMSE'].rank(ascending=True).astype(int)
        df['Correlation'] = df['Correlation'].rank(ascending=False).astype(int)
        df['NSE'] = df['NSE'].rank(ascending=False).astype(int)
        df['KGE'] = df['KGE'].rank(ascending=False).astype(int)
        df['Taylor_Skill_Score'] = df['Taylor_Skill_Score'].rank(ascending=False).astype(int)

        # 将当前 sheet 保存到新的 Excel 文件中
        df.to_excel(writer, sheet_name=sheet_name, index=False)
