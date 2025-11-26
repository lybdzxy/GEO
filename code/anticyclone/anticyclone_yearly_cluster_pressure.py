import pandas as pd
import matplotlib.pyplot as plt
import pymannkendall as mk
import palettable

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取CSV文件
df = pd.read_csv("merged_outer_new.csv")

# 将反气旋类别编号从 0-5 转换为 1-6
df.iloc[:, 19] += 1

# 选取相关列：年份、反气旋类别、中心气压
df_filtered = df.iloc[:, [1, 19, 6]]  # 1-年份, 19-类别, 6-中心气压

# 进行分组统计
result = df_filtered.groupby([df_filtered.columns[1], df_filtered.columns[0]])[
    df_filtered.columns[2]].mean().reset_index()

# 重命名列名
result.columns = ["Category", "Year", "Average Pressure"]

# 创建颜色列表
colors = palettable.colorbrewer.qualitative.Set1_6.mpl_colors

# 设置显著性水平
alpha = 0.05

# 创建一个图形，按照类别分组绘制不同的折线
plt.figure(figsize=(12, 8))

for i, category in enumerate(result["Category"].unique()):
    category_data = result[result["Category"] == category]
    pressures = category_data["Average Pressure"].values
    years = category_data["Year"].values

    # 执行 Mann-Kendall 趋势检验
    trend, h, p, _, _, _, _, _, _ = mk.original_test(pressures)

    # 设置线条粗细
    linewidth = 2 if h else 1

    # 绘制折线
    plt.plot(years, pressures, label=f"类别 {category.astype(int)}", color=colors[i], linewidth=linewidth)

# 设置图表标题和标签
plt.title("各聚类反气旋平均气压随年份变化", fontsize=14)
plt.xlabel("年份", fontsize=12)
plt.ylabel("平均气压 (hPa)", fontsize=12)

# 添加图例
plt.legend(title="反气旋聚类 ", title_fontsize=12, fontsize=10)

# 显示图表
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
