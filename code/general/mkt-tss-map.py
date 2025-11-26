import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from mpl_toolkits.axes_grid1 import AxesGrid
from cartopy.mpl.geoaxes import GeoAxes  # 导入GeoAxes，用于创建地理坐标轴
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter  # 导入LongitudeFormatter和LatitudeFormatter，用于格式化经纬度标签
import cartopy.feature as cfeature  # 导入cartopy.feature库，用于添加地理特征（如海岸线、河流等）
from matplotlib import rcParams  # 导入rcParams，用于配置matplotlib参数
import cmaps

ssp_values = [126, 245, 370, 585]
index_values = ['prcptot', 'r95p', 'r99p', 'r95ptot', 'r99ptot', 'sdii', 'rx1day', 'rx5day', 'r10mm', 'r20mm', 'cwd']

# 创建一个figure
fig = plt.figure(figsize=(44, 16))

# 创建一个AxesGrid，指定行数和列数，以及高度比例
axes_class = (GeoAxes, dict(map_projection=ccrs.PlateCarree()))
axgr = AxesGrid(fig, 111, axes_class=axes_class,
                nrows_ncols=(len(ssp_values), len(index_values)),
                axes_pad=0.05,
                cbar_location='right',
                cbar_mode='single',
                cbar_pad=0.1,
                cbar_size='3%',
                label_mode='')

for i, ssp in enumerate(ssp_values):
    for j, index in enumerate(index_values):
        # 打开NetCDF文件
        datapath = f'E:/GEO/result/{ssp}{index}_mktest.nc'
        data = xr.open_dataset(datapath)
        slope = data['slope']

        # 获取降水数据、经度和纬度
        lon = data['lon']
        lat = data['lat']

        # 创建一个子图
        ax = axgr[i*len(index_values)+j]

        # 绘制降水数据的轮廓填充
        cf = ax.pcolormesh(lon, lat, slope.T, cmap=cmaps.GMT_panoply, shading='auto')

        # 使用 cartopy.feature 中的 coastlines 函数绘制海岸线
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')

        # 从Shapefile文件中读取中国国界和省界数据
        china = shpreader.Reader('E:/GEO/geodata/bou2_4l.dbf').geometries()

        # 绘制中国国界、省界等
        for geom in china:
            ax.add_geometries([geom], ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.5, zorder=1)

# 添加行标题和列标题
for i, ssp in enumerate(ssp_values):
    ax = axgr[i*len(index_values)]
    ax.set_ylabel(f'SSP {ssp}', fontsize=12)
    ax.yaxis.set_label_coords(-0.2, 0)

for j, index in enumerate(index_values):
    axgr[j].set_title(index, fontsize=12)

# 去除子图的标题
plt.subplots_adjust(top=0.9)

plt.colorbar(cf,cax=axgr.cbar_axes[0])  # 绘制色标

# 显示图形
plt.savefig('E:/GEO/result/slope.png',dpi=100)