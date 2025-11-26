import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
from matplotlib.colors import BoundaryNorm

indices = ['prcptot', 'r95p', 'r99p', 'sdii', 'rx1day']
ssps = [126, 585]
for ssp in ssps:
    for index in indices:
        if ssp==126:
            boundary = (-99,-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6,99)
        else:
            boundary = (-99,-3.2, -2.4, -1.6, -0.8, 0, 0.8, 1.6, 2.4, 3.2,99)
        colors =[(103,0,31),
                (178,24,43),
                (214,96,77),
                (244,165,130),
                (253,219,199),
                (209,229,240),
                (146,197,222),
                (67,147,195),
                (33,102,172),
                (5,48,97)]

        # 根据颜色列表创建颜色映射
        scaled_colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]
        custom_cmap = ListedColormap(scaled_colors)

        norm = BoundaryNorm(boundary, len(scaled_colors))

        dataset_path = f'E:/GEO/result/ecm/{index}_tos_{ssp}.nc'
        dataset = xr.open_dataset(dataset_path)

        # 获取降水数据、经度和纬度
        lon = dataset['lon']
        lat = dataset['lat']
        data = dataset['rightPattern']

        # 创建一个figure
        fig = plt.figure(figsize=(12, 8))
        proj = ccrs.PlateCarree()  # 创建投影
        ax = fig.subplots(1, 1, subplot_kw={'projection': proj})

        # 中国经纬度范围
        region = [90, 140, 15, 55]
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

        # 绘制降水数据的轮廓填充
        cf = ax.pcolormesh(lon.values, lat.values, data[0, :, :].T, cmap=custom_cmap, norm=norm)

        # 从Shapefile文件中读取中国国界和省界数据
        china = shpreader.Reader('E:/GEO/geodata/bou2_4l.dbf').geometries()

        # 绘制中国国界、省界等
        for geom in china:
            ax.add_geometries([geom], ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.5, zorder=1)

        ##添加南海
        sub_ax = fig.add_axes([0.70, 0.20, 0.20, 0.20], projection=proj)
        sub_ax.set_extent([105, 125, 0, 25], crs=ccrs.PlateCarree())
        sub_ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
        sub_ax.pcolormesh(lon.values, lat.values, data[0, :, :].T, cmap=custom_cmap, norm=norm)
        sub_ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=1)
        china = shpreader.Reader('E:/GEO/geodata/bou2_4l.dbf').geometries()
        for geom in china:
            sub_ax.add_geometries([geom], ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.5, zorder=1)

        plt.savefig(f'E:/GEO/result/ecm/{index}_tos_{ssp}.png', dpi=600, bbox_inches='tight')
        plt.close()
