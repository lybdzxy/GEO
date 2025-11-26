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

def mk_map(index,ssp):
    colors = [(215,48,39), (255,255,191), (69,117,180)]
    scaled_colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]
    custom_cmap = ListedColormap(scaled_colors)
    labels = ['decreasing', 'no significant', 'increasing']

    # 创建一个figure
    fig = plt.figure(figsize=(12, 8))
    proj=ccrs.PlateCarree()  #创建投影
    ax=fig.subplots(1,1,subplot_kw={'projection':proj})

    #中国经纬度范围
    region = [70,140,15,55]
    ax.set_extent(region,crs=proj)

    # 设置地图属性:加载国界、海岸线、河流、湖泊
    #ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.8, zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=1)
    #ax.add_feature(cfeature.RIVERS.with_scale('50m'), zorder=1)
    #ax.add_feature(cfeature.LAKES.with_scale('50m'), zorder=1)

    # 设置网格点属性
    gl = ax.gridlines(ylocs=np.arange(region[2], region[3] + 10, 10), xlocs=np.arange(region[0], region[1] + 10, 10),
                      draw_labels=True, linestyle='--', alpha=0.7)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # 打开NetCDF文件
    datapath = f'E:/GEO/result/qpm/cut/{ssp}{index}_mktest.nc'
    data = xr.open_dataset(datapath)
    trend = data['trend']

    # 获取降水数据、经度和纬度
    lon = data['lon']
    lat = data['lat']

    # 绘制降水数据的轮廓填充
    cf = ax.pcolormesh(lon, lat, trend, cmap=custom_cmap, shading='auto')

    # 添加海岸线
    #ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')

    # 从Shapefile文件中读取中国国界和省界数据
    china = shpreader.Reader('E:/GEO/geodata/bou2_4l.dbf').geometries()

    # 绘制中国国界、省界等
    for geom in china:
        ax.add_geometries([geom], ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.5, zorder=1)

    ##添加南海
    sub_ax = fig.add_axes([0.70, 0.20, 0.20, 0.20], projection=proj)
    sub_ax.set_extent([105, 125, 0, 25], crs=ccrs.PlateCarree())
    sub_ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
    sub_ax.pcolormesh(lon, lat, trend, cmap=custom_cmap, shading='auto')
    sub_ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=1)
    china = shpreader.Reader('E:/GEO/geodata/bou2_4l.dbf').geometries()
    for geom in china:
        sub_ax.add_geometries([geom], ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.5, zorder=1)

    ''''# 设置标题
    ax.set_title(f'SSP {ssp} - {index}', fontsize=12)'''

    '''# 添加colorbar
    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.05)
    cbar.set_ticks([-1, 0, 1])
    cbar.ax.set_yticklabels(labels)
    '''
    # 显示图形
    plt.savefig(f'E:/GEO/result/qpm/map/{ssp}{index}_mktest_trend.png', dpi=600, bbox_inches='tight')

def main():
    # 定义索引和SSP值
    indices = ['prcptot', 'r95p', 'r99p', 'r95ptot', 'r99ptot', 'sdii', 'rx1day', 'r10mm', 'r20mm']
    ssps = ['126', '245', '370', '585']
    for index in indices:
        for ssp in ssps:
            mk_map(index,ssp)

if __name__ == '__main__':
    main()