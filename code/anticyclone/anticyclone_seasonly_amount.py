import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymannkendall as mk

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取CSV文件
df = pd.read_csv("trajectories_fin.csv")


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
df = df[~((df['Adjusted_Year'] == 1959) | ((df['Adjusted_Year'] == max_year) & (df['Season'] == 'Winter')))]

# 选取相关列：调整后的年份、季节、中心气压、trajectory_id
df_filtered = df[['Adjusted_Year', 'Season', 'center_pressure', 'trajectory_id']]

# 统计每年每个季节的反气旋数量
season_trajectory_count = df_filtered.groupby(['Adjusted_Year', 'Season'])['trajectory_id'].nunique().reset_index()

# 重命名列名
season_trajectory_count.columns = ['Year', 'Season', 'Cyclone Count']

# 定义季节映射（用于中文显示）
season_translation = {
    'Spring': '春天',
    'Summer': '夏天',
    'Fall': '秋天',
    'Winter': '冬天'
}

# 定义自定义颜色（春天绿色、夏天红色、秋天紫色、冬天蓝色）
color_map = {
    'Spring': 'green',
    'Summer': 'red',
    'Fall': 'purple',
    'Winter': 'blue'
}

# 创建一个图形，按照季节分组绘制不同的折线
plt.figure(figsize=(6.496, 4), dpi=300)

# 按季节顺序绘制折线
for season in ['Spring', 'Summer', 'Fall', 'Winter']:
    season_data = season_trajectory_count[season_trajectory_count["Season"] == season]
    counts = season_data["Cyclone Count"].values
    mean = counts.mean()
    print(f'{season}:{mean}')
    years = season_data["Year"].values

    # 执行 Mann-Kendall 趋势检验
    trend, h, p, _, _, _, _, _, _ = mk.original_test(counts)

    # 设置线条粗细
    linewidth = 2 if h else 1

    # 绘制折线，使用自定义颜色，并用中文标注
    chinese_season = season_translation[season]
    plt.plot(years, counts, label=f"{chinese_season} (p={p:.3f})", color=color_map[season], linewidth=linewidth)

# 设置图表标题和标签（使用中文）
plt.title("北半球温带反气旋季节数量随时间变化", fontsize=12)
plt.xlabel("时间（年）", fontsize=9)
plt.ylabel("反气旋数量", fontsize=9)

# 添加图例，使用中文
plt.legend(fontsize=9)

# 显示图表
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('E:/GEO/result/anticyclone/amount_seasonly.eps')
plt.show()
