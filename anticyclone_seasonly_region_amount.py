import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv("trajectories_fin_new.csv")

# 创建日期列
df['Date'] = pd.to_datetime(df[['year', 'month']].assign(DAY=1))

# 定义季节映射函数
def get_season(date):
    year = date.year
    month = date.month
    if month == 12:
        return ('Winter', year)  # 12月属于当年的冬季
    elif month in [1, 2]:
        return ('Winter', year - 1)  # 1月和2月属于前一年的冬季
    elif month in [3, 4, 5]:
        return ('Spring', year)
    elif month in [6, 7, 8]:
        return ('Summer', year)
    elif month in [9, 10, 11]:
        return ('Fall', year)
    return ('Unknown', year)

# 应用季节映射
season_data = df['Date'].apply(get_season)
df['Season'] = [x[0] for x in season_data]
df['Adjusted_Year'] = [x[1] for x in season_data]

# 剔除1959年和最后一个冬季
max_year = df['Adjusted_Year'].max()
df = df[~((df['Adjusted_Year'] == 1959) | ((df['Adjusted_Year'] == max_year) & (df['Season'] == 'Winter')))]

# 区域名称调整
region_mapping = {
    'Europe': '亚欧大陆',
    'Asia': '亚欧大陆',
    'Arctic Ocean': '北冰洋',
    'North Atlantic Ocean': '北大西洋',
    'North America': '北美大陆',
    'North Pacific Ocean': '北太平洋'
}
df['region_name'] = df['region_name'].replace(region_mapping)

# 剔除不需要的区域
regions_to_drop = ['Baltic Sea', 'Africa', 'Mediterranean Region']
df = df[~df['region_name'].isin(regions_to_drop)]

# 为每个trajectory_id确定最频繁出现的地区
df_region = df.groupby(['trajectory_id', 'region_name']).size().reset_index(name='count')
df_region = df_region.loc[df_region.groupby('trajectory_id')['count'].idxmax()]

# 合并回原数据
df = pd.merge(df, df_region[['trajectory_id', 'region_name']], on='trajectory_id', how='left', suffixes=('', '_most_frequent'))
df = df.drop(columns=['region_name'], errors='ignore')
df = df.rename(columns={'region_name_most_frequent': 'region_name'})

# 统计每个季节、每个地区每年的反气旋数量
season_region_trajectory_count = df.groupby(['Adjusted_Year', 'Season', 'region_name'])['trajectory_id'].nunique().reset_index()
season_region_trajectory_count.columns = ['Year', 'Season', 'Region', 'Cyclone Count']

# 计算每个季节、每个地区的平均反气旋数量
average_counts = season_region_trajectory_count.groupby(['Season', 'Region'])['Cyclone Count'].mean().reset_index()

# 创建季节-地区矩阵
pivot_table = average_counts.pivot(index='Season', columns='Region', values='Cyclone Count').fillna(0)

# 定义季节和地区的顺序
seasons_order = ['Spring', 'Summer', 'Fall', 'Winter']
regions_order = ['亚欧大陆', '北冰洋', '北大西洋', '北美大陆', '北太平洋']

# 调整矩阵顺序
pivot_table = pivot_table.reindex(seasons_order).reindex(columns=regions_order).fillna(0)

# 打印矩阵
print("各季节各地区的平均反气旋数量矩阵：")
print(pivot_table)

# 保存矩阵到CSV文件
pivot_table.to_csv("season_region_average_cyclone_count_new.csv", encoding='utf-8-sig')