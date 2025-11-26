import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 9

# 1. 读取数据
df = pd.read_csv('trajectories_fin.csv')  # 若非逗号分隔，请加 sep='\t'

# 2. 计算每条轨迹每年平均中心气压
mean_press = (
    df
    .groupby(['year', 'trajectory_id'])['center_pressure']
    .mean()
    .reset_index(name='avg_pressure')
)

# 3. 计算全时段（1960–2024）的第95百分位阈值
overall_threshold = mean_press['avg_pressure'].quantile(0.95)
print(f"整体95%分位阈值：{overall_threshold:.2f}")

# 4. 按年提取大于该阈值的轨迹，计算数量和平均值
def yearly_stats(group):
    high_pressures = group[group['avg_pressure'] > overall_threshold]['avg_pressure']
    return pd.Series({
        'count': len(high_pressures),
        'mean': high_pressures.mean() if not high_pressures.empty else np.nan,
        'std': high_pressures.std() if not high_pressures.empty else np.nan
    })

stats_by_year = (
    mean_press
    .groupby('year')
    .apply(yearly_stats)
    .reset_index()
)

# 5. 绘图：高压轨迹数量
plt.figure(figsize=(10, 4))
plt.plot(stats_by_year['year'], stats_by_year['count'], marker='o', label='数量')
plt.title('每年中心气压超过95%阈值的轨迹数量')
plt.xlabel('年份')
plt.ylabel('轨迹数量')
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. 绘图：高压轨迹平均气压及标准差范围
plt.figure(figsize=(10, 4))
mean_vals = stats_by_year['mean']
std_vals = stats_by_year['std']
years = stats_by_year['year']

plt.plot(years, mean_vals, marker='o', color='b', label='平均气压')
plt.fill_between(years, mean_vals - std_vals, mean_vals + std_vals, color='blue', alpha=0.3, label='±1 标准差')
plt.title('每年高中心气压轨迹的平均值及标准差')
plt.xlabel('年份')
plt.ylabel('平均气压 (hPa)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
