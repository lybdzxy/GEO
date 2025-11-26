import pandas as pd
sheet_names = ['prcptot', 'r95p', 'r99p', 'sdii', 'rx1day']
df_path = f'E:/GEO/etccdi/qpm/multi_year_rank.xlsx'
output_path = f'E:/GEO/etccdi/qpm/mul_rwb.xlsx'
with pd.ExcelWriter(output_path) as writer:
    for sheet in sheet_names:
        df = pd.read_excel(df_path, sheet_name=sheet)
        df['Si'] = df[['Correlation', 'NSE', 'KGE', 'Taylor_Skill_Score', 'NRMSE']].sum(axis=1)
        N = 12
        df['Ri'] = df['Si'].sum() / df['Si']
        Ri_sum = df['Ri'].sum()
        df['Wi'] = df['Ri'] / Ri_sum
        w_sum = df['Wi'].sum()
        df.to_excel(writer, sheet_name=sheet, index=False)
