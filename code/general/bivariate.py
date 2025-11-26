import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from mpl_toolkits.axes_grid1 import AxesGrid
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
from matplotlib import rcParams
import cmaps
from matplotlib.colors import ListedColormap
from matplotlib.colorbar import ColorbarBase
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import BoundaryNorm
import geopandas as gpd
import salem

district_path = f'E:/GEO/result/pca/NW.shp'
district = gpd.read_file(district_path)
data126_path = f'E:/GEO/result/new/cut/126prcptot_mktest.nc'
data585_path = f'E:/GEO/result/new/cut/585prcptot_mktest.nc'
data126 = xr.open_dataset(data126_path)
data126 = data126.salem.roi(shape=district)
data585 = xr.open_dataset(data585_path)
data585 = data585.salem.roi(shape=district)
trend126 = data126['trend']
trend585 = data585['trend']
trend126 = xr.where(trend126 == 1, 3, trend126)
trend126 = xr.where(trend126 == -1, 1, trend126)
trend126 = xr.where(trend126 == 0, 2, trend126)
trend585 = xr.where(trend585 == -1, 5, trend585)
trend585 = xr.where(trend585 == 0, 7, trend585)
trend585 = xr.where(trend585 == 1, 11, trend585)

trend = trend126*trend585

colors = [(170, 30, 33),
          (249, 202, 222),
          (255, 247, 149),
          (85, 47, 146),
          (230, 230, 230),
          (0, 91, 50),
          (134, 209, 212),
          (181, 227, 250),
          (0, 88, 168)]

boundary = (4,6,8,10.5,12,14.5,18,21.5,25,34)

# 根据颜色列表创建颜色映射
scaled_colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]
custom_cmap = ListedColormap(scaled_colors)

norm = BoundaryNorm(boundary, len(scaled_colors))

# 创建一个figure
fig = plt.figure(figsize=(12, 8))
proj = ccrs.PlateCarree()  # 创建投影
ax = fig.subplots(1, 1, subplot_kw={'projection': proj})

# 中国经纬度范围
#region = [70, 140, 15, 55]
region = [70, 110, 25, 50]
ax.set_extent(region, crs=proj)

# 设置地图属性:加载国界、海岸线、河流、湖泊
# ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.8, zorder=1)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=1)
# ax.add_feature(cfeature.RIVERS.with_scale('50m'), zorder=1)
# ax.add_feature(cfeature.LAKES.with_scale('50m'), zorder=1)

# 设置网格点属性
gl = ax.gridlines(ylocs=np.arange(region[2], region[3] + 10, 10), xlocs=np.arange(region[0], region[1] + 10, 10),
                  draw_labels=True, linestyle='--', alpha=0.7)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

# 获取降水数据、经度和纬度
lon = data126['lon']
lat = data126['lat']

# 绘制降水数据的轮廓填充
cf = ax.pcolormesh(lon, lat, trend, cmap=custom_cmap, norm=norm)

# 添加海岸线
# ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')

# 从Shapefile文件中读取中国国界和省界数据
china = shpreader.Reader('E:/GEO/geodata/bou2_4l.dbf').geometries()

# 绘制中国国界、省界等
for geom in china:
    ax.add_geometries([geom], ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.5, zorder=1)
'''# 添加colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="100%", pad=0.5, projection=proj)
cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', cax=cax, boundaries=boundary)
cbar.ax.set_aspect(1 / 20)
cbar.ax.set_xticks(boundary)'''
'''##添加南海
sub_ax = fig.add_axes([0.70, 0.20, 0.20, 0.20], projection=proj)
sub_ax.set_extent([105, 125, 0, 25], crs=ccrs.PlateCarree())
sub_ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
sub_ax.pcolormesh(lon, lat, trend, cmap=custom_cmap, shading='auto')
sub_ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=1)
china = shpreader.Reader('E:/GEO/geodata/bou2_4l.dbf').geometries()
for geom in china:
    sub_ax.add_geometries([geom], ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.5, zorder=1)'''

''''# 设置标题
ax.set_title(f'SSP {ssp} - {index}', fontsize=12)'''

'''# 添加colorbar
cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.05)
cbar.set_ticks([-1, 0, 1])
cbar.ax.set_yticklabels(labels)
'''
'''# 显示图形
plt.savefig(f'E:/GEO/result/qpm/map/{ssp}{index}_mktest_trend.png', dpi=600, bbox_inches='tight')'''
plt.show()