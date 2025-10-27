import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
from cartopy.mpl.geoaxes import GeoAxes
import matplotlib.path as mpath
import numpy as np

# 读取数据
data_path = 'E:/GEO/pyproject/traj_paths_2.5deg.nc'
data = xr.open_dataset(data_path)
annual_mean = data['traj_path'].mean(dim='year')

# 将值为0的部分设为NaN
annual_mean = annual_mean.where(annual_mean > 0)
annual_mean.to_netcdf('annual_mean.nc')
# 经纬度
lons = annual_mean['lon']
lats = annual_mean['lat']

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置 WGS84 投影
proj = ccrs.PlateCarree()  # WGS84 经纬度投影
pc = ccrs.PlateCarree()

# 绘图
fig, ax = plt.subplots(figsize=(6.496, 7), subplot_kw={'projection': proj})
ax.set_extent([60, 130, 30, 70], crs=pc)
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
# 分层设色边界
boundaries = [0, 1, 2, 3, 4, 5, annual_mean.max()]
cmap = plt.get_cmap('RdYlBu_r', len(boundaries) - 1)  # 选择ColorBrewer颜色映射
norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap.N, clip=False)

# 绘制主图
mesh = ax.pcolormesh(lons, lats, annual_mean,
                     transform=pc,
                     cmap=cmap,
                     norm=norm,
                     shading='auto',
                     zorder=0)

# 色带设置
ticks = boundaries[1:-1]  # 去掉最小值和最大值
cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal',
                    pad=0.05, shrink=0.8, boundaries=boundaries, ticks=ticks)
cbar.set_label('年均频次（次/年）', labelpad=20)

# 添加网格线
ax.gridlines(color='C7', lw=1, ls=':', draw_labels=True, rotate_labels=False, ylocs=[40, 60, 80])

# 标题与保存
plt.title('1960-2024年反气旋年均频次分布',
          pad=20, fontsize=12, weight='bold')
# plt.savefig('E:/GEO/result/anticyclone/frequency.eps', dpi=300)
plt.show()
