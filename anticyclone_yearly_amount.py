import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymannkendall as mk
from palettable.colorbrewer.qualitative import Set1_5  # 导入Set1_5配色方案

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取CSV文件（假设已有）
df = pd.read_csv("trajectories_fin_new.csv")  # 修正文件名，移除®

# 打印列名检查（保持原有）
print([repr(col) for col in df.columns])

# 去除列名中的引号和空格
df.columns = df.columns.str.replace("'", "").str.strip()

# 检查合并后的 df
print(df.head())

# 创建日期列
df['Date'] = pd.to_datetime(df[['year', 'month']].assign(DAY=1))

# 季节映射函数
def get_season(date):
    year = date.year
    month = date.month
    if month in [12]:
        return ('Winter', year)
    elif month in [1, 2]:
        return ('Winter', year - 1)
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

# 剔除第一个和最后一个冬季
max_year = df['Adjusted_Year'].max()
print(max_year)
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

# 打印列名确认
print(df.columns)

# 统计每个地区每年反气旋的数量
annual_region_trajectory_count = df.groupby(['Adjusted_Year', 'region_name'])['trajectory_id'].nunique().reset_index()
annual_region_trajectory_count.columns = ['Year', 'Region', 'Cyclone Count']

# 按地区分列输出 CSV（宽格式）
pivot_table = annual_region_trajectory_count.pivot(index='Year', columns='Region', values='Cyclone Count').reset_index()
pivot_table.to_csv("annual_region_trajectory_amount_pivoted_new.csv", index=False, encoding='utf-8-sig')

# 绘制按地区折线图并显示 p 值
plt.figure(figsize=(6.496, 4), dpi=300)

regions = annual_region_trajectory_count['Region'].unique()
colors = Set1_5.colors  # 使用 Set1_5 配色方案
# 将 RGB 值转换为十六进制颜色字符串
colors_hex = ['#{:02x}{:02x}{:02x}'.format(*color) for color in colors]

# 为每个区域计算 p 值并绘制
for i, region in enumerate(regions):
    region_data = annual_region_trajectory_count[annual_region_trajectory_count['Region'] == region]
    counts = region_data['Cyclone Count'].values

    # 执行 Mann-Kendall 趋势检验
    trend, h, p, _, _, _, _, _, _ = mk.original_test(counts)

    # 绘制折线，标签包含 p 值
    plt.plot(region_data['Year'], region_data['Cyclone Count'], label=f"{region} (p={p:.3f})", color=colors_hex[i])

# 设置图表标题和标签
plt.title("北半球各地区温带反气旋数量随时间变化", fontsize=12)
plt.xlabel("时间（年）", fontsize=9)
plt.ylabel("反气旋数量", fontsize=9)

# 添加图例
plt.legend(fontsize=9)

# 显示图表
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('E:/GEO/result/anticyclone/amount_region_new.eps')
plt.show()

# 整体趋势分析
annual_trajectory_count = df.groupby(['Adjusted_Year'])['trajectory_id'].nunique().reset_index()
annual_trajectory_count.columns = ['Year', 'Cyclone Count']

plt.figure(figsize=(6.496, 4), dpi=300)
counts = annual_trajectory_count["Cyclone Count"].values
years = annual_trajectory_count["Year"].values

trend, h, p, _, _, _, _, _, _ = mk.original_test(counts)
linewidth = 2 if h else 1

plt.plot(years, counts, label=f"反气旋数量 (p={p:.3f})", color='blue', linewidth=linewidth)
plt.title("北半球温带反气旋数量随时间变化", fontsize=12)
plt.xlabel("时间（年）", fontsize=9)
plt.ylabel("反气旋数量", fontsize=9)
plt.legend(fontsize=9)
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('E:/GEO/result/anticyclone/amount_yearly_new.eps')
plt.show()

# 保存整体反气旋数量数据
annual_trajectory_count.to_csv("annual_trajectory_amount_new.csv", index=False, encoding='utf-8-sig')
