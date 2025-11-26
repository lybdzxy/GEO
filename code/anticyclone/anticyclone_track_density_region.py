import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize
from matplotlib import cm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取 CSV 文件（假设文件名为 data.csv）
df = pd.read_csv("trajectories_filtered.csv")

# 选择 Y_bar < 70 的数据
filtered_data = df[df['Y_bar'] < 70]  # 假设 'Y_bar' 为需要筛选的列名

# 提取经纬度数据
lons = filtered_data.iloc[:, 2].values  # 假设经度在第三列
lats = filtered_data.iloc[:, 3].values  # 假设纬度在第四列
ids = filtered_data.iloc[:, 0].values  # 假设轨迹ID在第一列

# 将经度从 [0, 360] 转换到 [-180, 180]
lons = (lons + 180) % 360 - 180

# 创建投影转换器，设置中央经度为 180°
proj = ccrs.PlateCarree(central_longitude=180)

# 设置六边形网格的半径（单位：度）
hex_radius = 2.5

# 创建一个网格坐标范围
lon_bins = np.arange(-180, 180 + hex_radius, hex_radius)
lat_bins = np.arange(20, 90 + hex_radius, hex_radius)

# 创建一个空的矩阵，用于存储轨迹密度
density_grid = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1))

# 将轨迹点映射到对应的网格
for lon, lat in zip(lons, lats):
    # 获取轨迹点所在的网格索引
    lon_idx = np.digitize(lon, lon_bins) - 1
    lat_idx = np.digitize(lat, lat_bins) - 1

    # 判断索引是否有效
    if 0 <= lon_idx < len(lon_bins) - 1 and 0 <= lat_idx < len(lat_bins) - 1:
        density_grid[lat_idx, lon_idx] += 1  # 增加该网格内的轨迹点数

# 计算总月数（1960年1月1日到2024年12月31日）
start_date = pd.to_datetime('1960-01-01')
end_date = pd.to_datetime('2024-12-31')
total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month + 1)

# 计算月均轨迹密度
monthly_density_grid = density_grid / total_months

# 创建极地投影绘图
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': proj})

# 添加地图特征
ax.set_extent([-180, 180, 20, 90], crs=ccrs.PlateCarree())  # 只显示北纬 20° 及以上
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

# 绘制六边形网格
for i, lat in enumerate(lat_bins[:-1]):
    for j, lon in enumerate(lon_bins[:-1]):
        # 计算六边形中心的经纬度
        hex_lon = lon + hex_radius / 2
        hex_lat = lat + hex_radius / 2

        # 计算六边形的六个顶点坐标
        angles = np.linspace(0, 2 * np.pi, 7)
        hex_lons = hex_lon + hex_radius * np.cos(angles)
        hex_lats = hex_lat + hex_radius * np.sin(angles)

        # 绘制六边形
        ax.plot(hex_lons, hex_lats, color=cm.Reds(Normalize()(monthly_density_grid[i, j])))

# 添加颜色条
cbar = plt.colorbar(
    cm.ScalarMappable(norm=Normalize(vmin=np.min(monthly_density_grid), vmax=np.max(monthly_density_grid)),
                      cmap='Reds'), ax=ax, orientation='horizontal', pad=0.05)
cbar.set_label('月均轨迹密度')

# 添加网格和标题
ax.gridlines(draw_labels=False, linestyle="--", alpha=0.5)
ax.set_title("北半球温带反气旋月均轨迹密度六边形网格图", fontsize=14)

# 显示图形
plt.show()
