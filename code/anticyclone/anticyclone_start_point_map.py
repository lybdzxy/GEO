import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 16

# 设置处理的是“生成”点
n = 8
name = '生成'

# 读取 CSV 文件
df = pd.read_csv("trajectory_statistics_fin.csv")

# 提取经纬度数据
lons = df.iloc[:, n].values
lats = df.iloc[:, n + 1].values

# 创建投影
proj = ccrs.NorthPolarStereo()
pc = ccrs.PlateCarree()

# 经纬度转换为 NorthPolarStereo 坐标
x, y = proj.transform_points(pc, np.array(lons), np.array(lats))[:, :2].T

# 创建极地投影图
fig, ax = plt.subplots(figsize=(6.496, 7), subplot_kw={'projection': proj})
ax.set_extent([-180, 180, 20, 90], crs=pc)
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

# 绘制散点图
ax.scatter(x, y, s=1, color='red', alpha=0.5, transform=proj)

# 添加经纬网
ax.gridlines(color='C7', lw=1, ls=':', draw_labels=True, rotate_labels=False, ylocs=[40, 60, 80])

# 设置极地绘图边界
def polarCentral_set_latlim(lat_lims, ax):
    ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]], ccrs.PlateCarree())
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

polarCentral_set_latlim((20, 90), ax)

# 保存图像
# plt.savefig(f'E:/GEO/result/anticyclone/{name}_scatter.png', dpi=600)
plt.show()
