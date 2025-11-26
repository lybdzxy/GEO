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

# 选取相关列：年份、反气旋类别、轨迹ID、反气旋时长（第14列）
df_filtered = df.iloc[:, [1, 19, 0, 13]]  # 1-年份, 19-类别, 0-轨迹ID, 13-时长

# 按照年份和反气旋类别统计每年每个类别的反气旋时长均值
category_duration = df_filtered.groupby([df_filtered.columns[1], df_filtered.columns[0]])[df_filtered.columns[3]].mean().reset_index()

# 重命名列名
category_duration.columns = ["Category", "Year", "Average Duration"]

# 创建颜色列表
colors = palettable.colorbrewer.qualitative.Set1_6.mpl_colors

# 设置显著性水平
alpha = 0.05

# 创建一个图形，按照类别分组绘制不同的折线
plt.figure(figsize=(12, 8))

for i, category in enumerate(category_duration["Category"].unique()):
    category_data = category_duration[category_duration["Category"] == category]
    durations = category_data["Average Duration"].values
    years = category_data["Year"].values

    # 执行 Mann-Kendall 趋势检验
    trend, h, p, _, _, _, _, _, _ = mk.original_test(durations)

    # 设置线条粗细
    linewidth = 2 if h else 1

    # 绘制折线
    plt.plot(years, durations, label=f"类别 {category.astype(int)}", color=colors[i], linewidth=linewidth)

    # 在图例中注明显著性
    if h:
        plt.text(years[-1], durations[-1], f"p={p:.3f}*",
                 color=colors[i], fontsize=10, verticalalignment="bottom")

# 设置图表标题和标签
plt.title("各聚类反气旋时长随年份变化", fontsize=14)
plt.xlabel("年份", fontsize=12)
plt.ylabel("反气旋时长 (小时)", fontsize=12)  # 修改为小时

# 添加图例
plt.legend(title="反气旋聚类 ", title_fontsize=12, fontsize=10)

# 显示图表
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
