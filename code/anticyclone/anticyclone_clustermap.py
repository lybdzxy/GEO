import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import palettable
from matplotlib.colors import ListedColormap

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取数据
df = pd.read_csv("k-means_result_fin.csv")

# 提取经纬度和聚类编号数据
lons = df.iloc[:, 1].values  # 假设经度在第二列
lats = df.iloc[:, 2].values  # 假设纬度在第三列
clu = df.iloc[:, 13].values.astype(int) +1 # 确保聚类编号是整数

# 获取唯一的聚类编号，并排序
unique_clusters = np.unique(clu)
num_clusters = len(unique_clusters)

# 选择 palettable 颜色
colors = palettable.colorbrewer.qualitative.Set1_6.mpl_colors
cmap = ListedColormap(colors[:num_clusters])  # 只取前 num_clusters 个颜色
norm = mcolors.BoundaryNorm(boundaries=np.arange(num_clusters + 1) + 0.5, ncolors=num_clusters)

# 创建投影转换器
proj = ccrs.NorthPolarStereo()
pc = ccrs.PlateCarree()

# 将经纬度转换为 NorthPolarStereo 投影坐标
x, y = proj.transform_points(pc, np.array(lons), np.array(lats))[:, :2].T

# 创建极地投影绘图
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': proj})

# 添加地图特征
ax.set_extent([-180, 180, 20, 90], crs=ccrs.PlateCarree())  # 只显示北纬 20° 及以上
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

# 绘制散点图
scatter = ax.scatter(x, y, c=clu, cmap=cmap, norm=norm, s=10, alpha=0.75)

# 创建颜色条
cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.05)
cbar.set_label('聚类')
cbar.set_ticks(unique_clusters)
cbar.set_ticklabels(unique_clusters)

# 添加网格
gl = ax.gridlines(draw_labels=False, linestyle="--", alpha=0.5)
gl.xlocator = plt.MultipleLocator(30)  # 经度每 30°
gl.ylocator = plt.MultipleLocator(10)  # 纬度每 10°


ax.set_title("北半球温带反气旋空间聚类分布图", fontsize=14)

plt.show()
