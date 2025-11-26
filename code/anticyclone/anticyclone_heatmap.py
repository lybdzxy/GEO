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
plt.rcParams['font.size'] = 16

# 读取 CSV 文件（假设文件名为 data.csv）
df = pd.read_csv("trajectories_zs.csv")
filtered_data = df

# 提取经纬度数据
lons = filtered_data.iloc[:, 2].values  # 假设经度在第二列
lats = filtered_data.iloc[:, 3].values  # 假设纬度在第三列
ids = filtered_data.iloc[:, 0].values  # 假设轨迹ID在第一列
date = filtered_data.iloc[:, 1].values

# 创建投影转换器
proj = ccrs.NorthPolarStereo()
pc = ccrs.PlateCarree()

# 将经纬度转换为 NorthPolarStereo 投影坐标
x, y = proj.transform_points(pc, np.array(lons), np.array(lats))[:, :2].T

# 创建极地投影绘图
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': proj})

# 添加地图特征
ax.set_extent([-180, 180, 20, 90], crs=ccrs.PlateCarree())  # 只显示北纬 30° 及以上
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

# 绘制热点图（在转换后的坐标上）
hb = sns.kdeplot(x=x, y=y, alpha=0.6, levels=20, cmap="Reds", ax=ax, fill=True)

# 获取热点图数据的最小值和最大值
min_val = np.min(hb.collections[0].get_array())
max_val = np.max(hb.collections[0].get_array())

# 创建一个 ScalarMappable 对象
sm = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(vmin=min_val, vmax=max_val))

# 绘制颜色条
cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05)
cbar.set_label('核密度')

# 添加网格和标题
ax.gridlines(color='C7', lw=1, ls=':', draw_labels=True, rotate_labels=False, ylocs=[40, 60, 80])

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

# 设置标题
# ax.set_title("北半球温带反气旋成熟点核密度空间分布图", fontsize=14)
#plt.savefig('E:/GEO/result/anticyclone/路径_kde.png', dpi=600)
plt.show()

# 创建颜色条单独保存为 PNG
fig_cbar, ax_cbar = plt.subplots(figsize=(8, 1))  # 创建一个新的图像和轴用于颜色条
cbar = plt.colorbar(sm, ax=ax_cbar, orientation='horizontal')
cbar.set_label('核密度')
#fig_cbar.savefig('E:/GEO/result/anticyclone/colorbar.png', dpi=600, bbox_inches='tight')
plt.close(fig_cbar)  # 关闭颜色条图像的fig
