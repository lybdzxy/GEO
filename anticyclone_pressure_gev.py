import pandas as pd
from lmoments3 import distr
from scipy.stats import genextreme
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

# 3. 计算每年 95% 分位数阈值
quantiles = (
    mean_press
    .groupby('year')['avg_pressure']
    .quantile(0.95)
    .reset_index(name='threshold')
)

# 4. 合并并筛选出每年气压最高 5% 的轨迹
top5 = (
    mean_press
    .merge(quantiles, on='year')
    .query('avg_pressure >= threshold')
    .sort_values(['year', 'avg_pressure'], ascending=[True, False])
)

# 5. 计算每年这 5% 轨迹的平均气压
yearly_top5_mean = (
    top5
    .groupby('year')['avg_pressure']
    .max()
    .reset_index(name='mean_of_top5')
    .sort_values('year')
)

print(yearly_top5_mean)

# 6. 按时间段分组
group1 = yearly_top5_mean[
    (yearly_top5_mean['year'] >= 1960) &
    (yearly_top5_mean['year'] <= 1992)
]
group2 = yearly_top5_mean[
    (yearly_top5_mean['year'] >= 1993) &
    (yearly_top5_mean['year'] <= 2024)
]

# 7. 拟合 GEV 分布（L-矩法）
data1 = group1['mean_of_top5'].values
data2 = group2['mean_of_top5'].values

gev_params_1 = distr.gev.lmom_fit(data1)
gev_params_2 = distr.gev.lmom_fit(data2)

print("1960–1989 残差极值分布参数：", gev_params_1)
print("1990–2024 残差极值分布参数：", gev_params_2)

# 如果需要单独访问 loc、scale、shape：
loc1, scale1, shape1 = gev_params_1['loc'], gev_params_1['scale'], gev_params_1['c']
loc2, scale2, shape2 = gev_params_2['loc'], gev_params_2['scale'], gev_params_2['c']
# 生成 x 值范围
min_val = min(data1.min(), data2.min())
max_val = max(data1.max(), data2.max())
x_values = np.linspace(min_val, max_val, 200)

# 计算 PDF
pdf1 = genextreme.pdf(x_values, shape1, loc=loc1, scale=scale1)
pdf2 = genextreme.pdf(x_values, shape2, loc=loc2, scale=scale2)

# 绘图
plt.figure()
plt.plot(x_values, pdf1)
plt.plot(x_values, pdf2)
plt.title('1960–1992 与 1993–2024 两时期 GEV 分布 PDF')
plt.xlabel('年度最大平均中心气压')
plt.ylabel('概率密度')
plt.legend(['1960–1992', '1993–2024'])
plt.show()
