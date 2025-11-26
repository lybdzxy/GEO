import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
from cartopy.mpl.geoaxes import GeoAxes
import matplotlib.path as mpath

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 9
# 读取 CSV 文件（假设文件名为 data.csv）
df = pd.read_csv("trajectories_zs.csv")

# 提取经纬度数据
lons = df.iloc[:, 2].values  # 假设经度在第3列
lats = df.iloc[:, 3].values  # 假设纬度在第4列
ids = df.iloc[:, 0].values  # 假设轨迹ID在第1列
pressure = df.iloc[:, 4].values  # 假设压力在第7列

# 创建投影转换器
proj = ccrs.NorthPolarStereo()
pc = ccrs.PlateCarree()

# 创建极地投影绘图
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': proj})
ax.set_extent([-180, 180, 20, 90], crs=ccrs.PlateCarree())  # 只显示北纬 20° 及以上
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

# 创建一个压力的颜色映射
cmap = plt.get_cmap('Reds')
norm = mcolors.BoundaryNorm(boundaries=[1025, 1030, 1035, 1040, 1045, 1050], ncolors=256)

# 创建一个字典来存储不同轨迹的点
trajectory_dict = {}

# 将相同ids的点连接成线
for i, traj_id in enumerate(ids):
    if traj_id not in trajectory_dict:
        trajectory_dict[traj_id] = {'lons': [], 'lats': [], 'stream': []}

    trajectory_dict[traj_id]['lons'].append(lons[i])
    trajectory_dict[traj_id]['lats'].append(lats[i])
    trajectory_dict[traj_id]['stream'].append(pressure[i])

# 绘制每条轨迹和点
for traj_id, traj_data in trajectory_dict.items():
    # 轨迹线
    ax.plot(traj_data['lons'], traj_data['lats'], color='black', linewidth=0.2, transform=ccrs.PlateCarree())

    # 绘制每个点，根据压力值选择颜色
    for lon, lat, pres in zip(traj_data['lons'], traj_data['lats'], traj_data['stream']):
        ax.scatter(lon, lat, c=pres, cmap=cmap, norm=norm, s=1, transform=ccrs.PlateCarree())

# 添加颜色条
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='horizontal', fraction=0.02, pad=0.04)
cbar.set_label('位势高度')

# 添加经纬网及标注
def polarCentral_set_latlim(lat_lims, ax):
    ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]], ccrs.PlateCarree())
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

# 设置绘制区域范围
polarCentral_set_latlim((20, 90), ax)
# 显示地图
plt.show()
