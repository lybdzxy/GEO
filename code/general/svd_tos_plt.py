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
            boundary = (-99,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,99)
        else:
            boundary = (-99,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,99)
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

        dataset_path = f'E:/GEO/result/ecm/tos_{index}_{ssp}.nc'
        dataset = xr.open_dataset(dataset_path)

        # 获取降水数据、经度和纬度
        lon = dataset['lon']
        lat = dataset['lat']
        data = dataset['leftPattern']

        # 创建一个figure
        fig = plt.figure(figsize=(12, 8))
        proj = ccrs.PlateCarree()  # 创建投影
        ax = fig.subplots(1, 1, subplot_kw={'projection': proj})

        # 经纬度范围
        region = [0, 359.5, -90, 90]
        ax.set_extent(region, crs=proj)

        # 设置地图属性:加载国界、海岸线、河流、湖泊
        # ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.8, zorder=1)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=1)
        # ax.add_feature(cfeature.RIVERS.with_scale('50m'), zorder=1)
        # ax.add_feature(cfeature.LAKES.with_scale('50m'), zorder=1)

        # 设置网格点属性
        gl = ax.gridlines(ylocs=np.arange(region[2], region[3] + 10, 30), xlocs=np.arange(region[0]-180, region[1]-180, 30),
                          draw_labels=True, linestyle='--', alpha=0.7)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        # 绘制降水数据的轮廓填充
        cf = ax.pcolormesh(lon.values, lat.values, data[0, :, :].T, cmap=custom_cmap, norm=norm)

        '''        # 从Shapefile文件中读取边界数据
        world = shpreader.Reader('E:/GEO/geodata/bou2_4l.dbf').geometries()

        # 绘制边界等
        for geom in world:
            ax.add_geometries([geom], ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.5, zorder=1)'''
        plt.savefig(f'E:/GEO/result/ecm/tos_{index}_{ssp}.png', dpi=600, bbox_inches='tight')
        plt.close()
