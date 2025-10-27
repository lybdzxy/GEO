import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymannkendall as mk
from palettable.colorbrewer.qualitative import Set1_5  # 导入Set1_5配色方案

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 读取CSV文件
df = pd.read_csv("trajectories_fin_new.csv")
df.columns = df.columns.str.replace("'", "").str.strip()

# 创建日期列（假设原数据有 'year' 和 'month' 列）
df['Date'] = pd.to_datetime(df[['year', 'month']].assign(DAY=1))

# 创建季节映射函数
def get_season(date):
    year = date.year
    month = date.month

    # 冬季：当年的12月属于当年的冬季，前一年的1月和2月属于前一年的冬季
    if month in [12]:
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

# 应用季节映射函数，获取季节和调整后的年份
season_data = df['Date'].apply(get_season)
df['Season'] = [x[0] for x in season_data]  # 提取季节
df['Adjusted_Year'] = [x[1] for x in season_data]  # 提取调整后的年份

# 剔除第一个冬季（1959年）和最后一个冬季
# 找到最大年份
max_year = df['Adjusted_Year'].max()

# 剔除1959年和最大年份的冬季数据
df = df[~((df['Adjusted_Year'] == 1959) | (df['Adjusted_Year'] == max_year) )]
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

# 重新处理每个 trajectory_id 对应的 region_name
df_region = df.groupby(['trajectory_id', 'region_name']).size().reset_index(name='count')
df_region = df_region.loc[df_region.groupby('trajectory_id')['count'].idxmax()]

# 合并回原数据
df = pd.merge(df, df_region[['trajectory_id', 'region_name']], on='trajectory_id', how='left',
              suffixes=('', '_most_frequent'))
df = df.drop(columns=['region_name'], errors='ignore')
df = df.rename(columns={'region_name_most_frequent': 'region_name'})
# 统计每个地区每年反气旋的数量
annual_region_trajectory_count = df.groupby(['Adjusted_Year', 'region_name'])['center_pressure'].mean().reset_index()
annual_region_trajectory_count.columns = ['Year', 'Region', 'Average Pressure']
# 按地区分列输出 CSV（宽格式）
pivot_table = annual_region_trajectory_count.pivot(index='Year', columns='Region', values='Average Pressure').reset_index()
pivot_table.to_csv("annual_region_trajectory_pressure_pivoted_new.csv", index=False, encoding='utf-8-sig')
# 创建一个图形，绘制整体折线
plt.figure(figsize=(6.496, 4), dpi=300)
regions = annual_region_trajectory_count['Region'].unique()
colors = Set1_5.colors  # 使用 Set1_5 配色方案
# 将 RGB 值转换为十六进制颜色字符串
colors_hex = ['#{:02x}{:02x}{:02x}'.format(*color) for color in colors]# 为每个区域计算 p 值并绘制
for i, region in enumerate(regions):
    region_data = annual_region_trajectory_count[annual_region_trajectory_count['Region'] == region]
    counts = region_data['Average Pressure'].values

    # 执行 Mann-Kendall 趋势检验
    trend, h, p, _, _, _, _, _, _ = mk.original_test(counts)

    # 绘制折线，标签包含 p 值
    plt.plot(region_data['Year'], region_data['Average Pressure'], label=f"{region} (p={p:.3f})", color=colors_hex[i])

# 设置图表标题和标签
plt.title("北半球各地区温带反气旋平均中心气压随时间变化", fontsize=12)
plt.xlabel("时间（年）", fontsize=9)
plt.ylabel("反气旋平均中心气压（hPa）", fontsize=9)

# 添加图例
plt.legend(fontsize=9)

# 显示图表
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('E:/GEO/result/anticyclone/pressure_region_new.eps')
plt.show()

# 整体趋势分析
annual_trajectory_count = df.groupby(['Adjusted_Year'])['center_pressure'].mean().reset_index()
annual_trajectory_count.columns = ['Year', 'Average Pressure']

plt.figure(figsize=(6.496, 4), dpi=300)
counts = annual_trajectory_count["Average Pressure"].values
years = annual_trajectory_count["Year"].values

trend, h, p, _, _, _, _, _, _ = mk.original_test(counts)
linewidth = 2 if h else 1

plt.plot(years, counts, label=f"反气旋平均中心气压 (p={p:.3f})", color='blue', linewidth=linewidth)
plt.title("北半球温带反气旋平均中心气压随时间变化", fontsize=12)
plt.xlabel("时间（年）", fontsize=9)
plt.ylabel("反气旋平均中心气压（hPa）", fontsize=9)
plt.legend(fontsize=9)
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('E:/GEO/result/anticyclone/pressure_yearly_new.eps')
plt.show()

# 保存整体反气旋数量数据
annual_trajectory_count.to_csv("annual_trajectory_pressure_new.csv", index=False, encoding='utf-8-sig')