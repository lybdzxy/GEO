import pandas as pd
import pymannkendall as mk
import matplotlib.pyplot as plt
from scipy.stats import linregress

ssp_values = ('126', '245', '370', '585', 'his')
indices = ('r99p', 'rx5day', 'prcptot', 'r20mm')
indices_unit = ('r99p (mm)', 'rx5day (mm)', 'prcptot (mm)', 'r20mm (day)')

# 创建一个子图的布局，调整图像大小
plt.figure(figsize=(20, 16))

plot_num = 1
# 循环处理每个文件并在子图中绘制图表
for ssp in ssp_values:
    for index in indices:
        file_path = f'E:/GEO/etccdi/{ssp}/{index}.xlsx'
        df = pd.read_excel(file_path, sheet_name='Sheet1')
        data = df['Mean Value']
        result = mk.original_test(data)

        # 创建子图
        plt.subplot(5, 4, plot_num)

        # 绘制数据趋势图
        plt.plot(df['Year'], data, marker='o', linestyle='-')

        # 添加线性回归趋势线
        slope, intercept, r_value, p_value, std_err = linregress(df['Year'], data)
        trendline = intercept + slope * df['Year']
        plt.plot(df['Year'], trendline, color='red', linestyle='--', label='Linear Regression Trend')

        # 添加标签和标题
        plt.xlabel('Year')

        # 显示趋势线的方向
        if result.trend == "increasing":
            trend_text = "Increasing Trend"
        elif result.trend == "decreasing":
            trend_text = "Decreasing Trend"
        else:
            trend_text = "No Significant Trend"

        plt.text(0.1, 0.85, f'Trend: {trend_text}\nSlope: {result.slope:.4f}',
                 transform=plt.gca().transAxes, fontsize=10)

        plot_num = plot_num + 1

# 添加行标题和列标题
for i, ssp in enumerate(ssp_values):
    plt.subplot(5, 4, i * 4 + 1)
    plt.ylabel(f'SSP {ssp}', fontsize=12)

for i, index in enumerate(indices_unit):
    plt.subplot(5, 4, i + 1)
    plt.title(index, fontsize=12)

# 去除子图的标题
plt.subplots_adjust(top=0.9)

# 保存图像
plt.savefig('trend_analysis.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()
