import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
from matplotlib.colors import ListedColormap

def mk_map(index, ssp, boundaries):
    # 定义自定义颜色和对应值范围
    colors = [(103 / 255, 0 / 255, 31 / 255),
              (178 / 255, 24 / 255, 43 / 255),
              (214 / 255, 96 / 255, 77 / 255),
              (244 / 255, 165 / 255, 130 / 255),
              (253 / 255, 219 / 255, 199 / 255),
              (209 / 255, 229 / 255, 240 / 255),
              (146 / 255, 197 / 255, 222 / 255),
              (67 / 255, 147 / 255, 195 / 255),
              (33 / 255, 102 / 255, 172 / 255),
              (5 / 255, 48 / 255, 97 / 255)]
    #boundaries = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25]

    # 创建一个figure
    fig = plt.figure(figsize=(12, 8))
    proj = ccrs.PlateCarree()  # 创建投影
    ax = fig.subplots(1, 1, subplot_kw={'projection': proj})

    # 中国经纬度范围
    region = [70, 140, 15, 55]
    ax.set_extent(region, crs=proj)

    # 设置地图属性:加载海岸线
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=1)

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
    slope = data['slope']

    # 获取降水数据、经度和纬度
    lon = data['lon']
    lat = data['lat']

    # 将数据限制在边界值范围内
    slope = np.clip(slope, boundaries[0], boundaries[-1])

    # 绘制降水数据的轮廓填充
    cf = ax.pcolormesh(lon, lat, slope, cmap=ListedColormap(colors), vmin=boundaries[0], vmax=boundaries[-1], shading='auto')

    # 从Shapefile文件中读取中国国界数据
    china = shpreader.Reader('E:/GEO/geodata/bou2_4l.dbf').geometries()

    # 绘制中国国界
    for geom in china:
        ax.add_geometries([geom], ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.5, zorder=1)

    #添加南海
    sub_ax = fig.add_axes([0.70, 0.20, 0.20, 0.20], projection=proj)
    sub_ax.set_extent([105, 125, 0, 25], crs=ccrs.PlateCarree())
    sub_ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
    sub_ax.pcolormesh(lon, lat, slope, cmap=ListedColormap(colors), vmin=-25, vmax=25, shading='auto')
    sub_ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=1)
    china = shpreader.Reader('E:/GEO/geodata/bou2_4l.dbf').geometries()
    for geom in china:
        sub_ax.add_geometries([geom], ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.5, zorder=1)

    '''# 添加colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="100%", pad=0.5, projection=proj)
    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', cax=cax, boundaries=boundaries)
    cbar.ax.set_aspect(1 / 20)
    cbar.ax.set_xticks(boundaries)'''
    '''# 单独输出colorbar到一个独立的图中
    fig_cb = plt.figure(figsize=(8, 1))
    ax_cb = fig_cb.add_subplot(111)
    cbar = plt.colorbar(cf, orientation='horizontal', cax=ax_cb, boundaries=boundaries)
    cbar.ax.set_aspect(1 / 20)
    cbar.ax.set_xticks(boundaries)
    plt.savefig(f'E:/GEO/result/qpm/map/{ssp}{index}_mktest_slope_colorbar.png', dpi=600, bbox_inches='tight')'''

    plt.savefig(f'E:/GEO/result/qpm/map/{ssp}{index}_mktest_slope.png', dpi=600, bbox_inches='tight')


def main():
    boundaries = {
        'prcptot': (-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25),
        'r95p': (-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25),
        'r99p': (-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10),
        'r95ptot': (-0.006, -0.0048, -0.0036, -0.0024, -0.0012, 0, 0.0012, 0.0024, 0.0036, 0.0048, 0.006),
        'r99ptot': (-0.006, -0.0048, -0.0036, -0.0024, -0.0012, 0, 0.0012, 0.0024, 0.0036, 0.0048, 0.006),
        'sdii': (-0.025, -0.020, -0.015, -0.010, -0.005, 0, 0.005, 0.010, 0.015, 0.020, 0.025),
        'rx1day': (-4.0, -3.2, -2.4, -1.6, -0.8, 0, 0.8, 1.6, 2.4, 3.2, 4.0),
        'r10mm': (-0.35, -0.28, -0.21, -0.14, -0.07, 0, 0.07, 0.14, 0.21, 0.28, 0.35),
        'r20mm': (-0.30, -0.24, -0.18, -0.12, -0.06, 0, 0.06, 0.12, 0.18, 0.24, 0.30)
    }
    # 定义索引和SSP值
    indices = ['prcptot', 'r95p', 'r99p', 'r95ptot', 'r99ptot', 'sdii', 'rx1day', 'r10mm', 'r20mm']
    ssps = ['126', '245', '370', '585']
    for index in indices:
        for ssp in ssps:
            mk_map(index, ssp, boundaries[index])


if __name__ == '__main__':
    main()
