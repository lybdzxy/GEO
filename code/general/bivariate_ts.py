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
from matplotlib.colors import BoundaryNorm


indices = ['prcptot', 'r95p', 'r99p', 'sdii', 'rx1day']
for index in indices:
    data126_path = f'E:/GEO/result/ecm/126{index}_mktest.nc'
    data585_path = f'E:/GEO/result/ecm/585{index}_mktest.nc'
    data126 = xr.open_dataset(data126_path)
    data585 = xr.open_dataset(data585_path)
    ts_126 = data126['slope']
    ts_585 = data585['slope']

    # 将数据一维化并剔除 NaN
    ts_126_flaten = ts_126.values[~np.isnan(ts_126.values)]
    ts_585_flaten = ts_585.values[~np.isnan(ts_585.values)]

    # 计算百分之5和95百分位数
    p5_126, p95_126 = np.percentile(ts_126_flaten, [1, 99])
    p5_585, p95_585 = np.percentile(ts_585_flaten, [1, 99])

    # 选取百分之5到95百分位数的数据
    ts_126_filtered = ts_126_flaten[(ts_126_flaten >= p5_126) & (ts_126_flaten <= p95_126)]
    ts_585_filtered = ts_585_flaten[(ts_585_flaten >= p5_585) & (ts_585_flaten <= p95_585)]

    # 计算标准差
    std_126 = np.std(ts_126_filtered)
    std_585 = np.std(ts_585_filtered)

    # 为 ts_126 和 ts_585 创建新的变量来存储分类结果
    classified_ts_126 = xr.full_like(ts_126, np.NAN)  # 初始化为全0数组
    classified_ts_585 = xr.full_like(ts_585, np.NAN)  # 初始化为全0数组

    # 对 ts_126 进行分类
    classified_ts_126 = xr.where((ts_126 < (0 - std_126)), 1, classified_ts_126)
    classified_ts_126 = xr.where((ts_126 < 0) & (ts_126 > (0 - std_126)), 2, classified_ts_126)
    classified_ts_126 = xr.where((ts_126 < (0 + 1.5 * std_126)) & (ts_126 > 0), 3, classified_ts_126)
    classified_ts_126 = xr.where((ts_126 > (0 + 1.5 * std_126)), 4, classified_ts_126)

    # 对 ts_585 进行分类
    classified_ts_585 = xr.where((ts_585 < (0 - std_585)), 1, classified_ts_585)
    classified_ts_585 = xr.where((ts_585 < 0) & (ts_585 > (0 - std_585)), 5, classified_ts_585)
    classified_ts_585 = xr.where((ts_585 < (0 + 1.5 * std_585)) & (ts_585 > 0), 30, classified_ts_585)
    classified_ts_585 = xr.where((ts_585 > (0 + 1.5 * std_585)), 150, classified_ts_585)

    # 最终的 trend 计算
    slope = (classified_ts_126 * classified_ts_585).T

    colors =[(242, 115, 0),
            (230, 200, 128),
            (149, 230, 125),
            (0, 136, 54),
            (242, 153, 145),
            (255, 231, 217),
            (217, 255, 230),
            (108, 217, 211),
            (230, 127, 182),
            (255, 216, 243),
            (222, 217, 255),
            (138, 173, 229),
            (241, 0, 139),
            (230, 128, 230),
            (188, 127, 230),
            (90, 78, 164),
            ]

    boundary = (0,1.5,2.5,3.5,4.5,7,12,17,25,50,70,100,130,200,350,500,700)

    # 根据颜色列表创建颜色映射
    scaled_colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]
    custom_cmap = ListedColormap(scaled_colors)

    norm = BoundaryNorm(boundary, len(scaled_colors))

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

    # 获取降水数据、经度和纬度
    lon = data126['level_0']
    lat = data126['level_1']

    # 绘制降水数据的轮廓填充
    cf = ax.pcolormesh(lon, lat, slope, cmap=custom_cmap, norm=norm, shading='goraud')

    # 添加海岸线
    # ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')

    # 从Shapefile文件中读取中国国界和省界数据
    china = shpreader.Reader('E:/GEO/geodata/bou2_4l.dbf').geometries()

    # 绘制中国国界、省界等
    for geom in china:
        ax.add_geometries([geom], ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.5, zorder=1)

    '''##添加南海
    sub_ax = fig.add_axes([0.70, 0.20, 0.20, 0.20], projection=proj)
    sub_ax.set_extent([105, 125, 0, 25], crs=ccrs.PlateCarree())
    sub_ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
    sub_ax.pcolormesh(lon, lat, trend, cmap=custom_cmap,norm=norm, shading='auto')
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
    # 显示图形
    plt.savefig(f'E:/GEO/result/ecm/pic/{index}_mktest_slope.png', dpi=600, bbox_inches='tight')
    plt.close()
