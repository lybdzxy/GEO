"""
郑州极端降水环流分析
Analysis of synoptic circulation for Zhengzhou extreme precipitation event (2021.7.20)

功能：
1. 500hPa/850hPa位势高度场和风场分析
2. 副热带高压、低涡等系统识别
3. 水汽通量和辐合分析

使用ERA5数据进行分析
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import os
from datetime import datetime

# ==================== 配置参数 ====================
# 数据路径配置 - 请根据实际情况修改
DATA_DIR = r'E:/ERA5/hourly'  # ERA5数据目录
OUTPUT_DIR = r'E:/GEO/pyproject/casestudy/zhengzhou'  # 输出目录

# 郑州坐标
ZHENGZHOU_LON = 113.65
ZHENGZHOU_LAT = 34.76

# 分析区域 [lon_min, lon_max, lat_min, lat_max]
ANALYSIS_REGION = [100, 130, 20, 45]

# 分析时间 (2021年7月20日)
ANALYSIS_DATE = datetime(2021, 7, 20, 12)  # UTC时间


def load_era5_data(data_dir, date, levels=[500, 850]):
    """
    加载ERA5数据
    
    参数:
        data_dir: 数据目录
        date: 分析日期
        levels: 需要的气压层 (hPa)
    
    返回:
        包含各变量的字典
    """
    year = date.year
    month = date.month
    day = date.day
    hour = date.hour
    
    # 构建文件路径 - 根据实际数据组织方式修改
    # 假设文件命名格式: ERA5_YYYY-MM-DD_pl_q.nc, ERA5_YYYY-MM-DD_pl_u.nc 等
    date_str = date.strftime('%Y-%m-%d')
    
    data = {}
    
    # 加载气压层数据
    try:
        # 尝试加载合并后的文件
        pl_path = os.path.join(data_dir, f'ERA5_{date_str}_pl.nc')
        if os.path.exists(pl_path):
            pl_data = xr.open_dataset(pl_path)
        else:
            # 分别加载各变量文件
            u_path = os.path.join(data_dir, f'ERA5_{date_str}_pl_u.nc')
            v_path = os.path.join(data_dir, f'ERA5_{date_str}_pl_v.nc')
            q_path = os.path.join(data_dir, f'ERA5_{date_str}_pl_q.nc')
            
            u_data = xr.open_dataset(u_path)
            v_data = xr.open_dataset(v_path)
            q_data = xr.open_dataset(q_path)
            
            pl_data = xr.merge([u_data, v_data, q_data])
    except FileNotFoundError as e:
        print(f"警告: 找不到气压层数据文件: {e}")
        return None
    
    # 加载地表数据 (表面气压)
    try:
        sp_path = os.path.join(data_dir, f'ERA5_{date_str}_sp.nc')
        if os.path.exists(sp_path):
            sp_data = xr.open_dataset(sp_path)
            data['sp'] = sp_data
    except FileNotFoundError:
        print("警告: 找不到地表气压数据")
    
    data['pl'] = pl_data
    
    return data


def calculate_geopotential_height(z):
    """
    将位势转换为位势高度 (gpm)
    
    参数:
        z: 位势 (m²/s²)
    
    返回:
        位势高度 (gpm)
    """
    g = 9.80665  # 重力加速度
    return z / g


def calculate_wind_speed(u, v):
    """计算风速"""
    return np.sqrt(u**2 + v**2)


def plot_geopotential_wind(z, u, v, level, ax, region, center_lon=113.65, center_lat=34.76):
    """
    绘制位势高度场和风场
    
    参数:
        z: 位势高度 (gpm)
        u, v: 风速分量 (m/s)
        level: 气压层 (hPa)
        ax: matplotlib轴对象
        region: 绘图区域
        center_lon, center_lat: 中心点坐标
    """
    lon = z.longitude.values if 'longitude' in z.dims else z.lon.values
    lat = z.latitude.values if 'latitude' in z.dims else z.lat.values
    
    # 设置地图范围
    ax.set_extent(region, crs=ccrs.PlateCarree())
    
    # 添加地图要素
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    
    # 创建网格
    lon2d, lat2d = np.meshgrid(lon, lat)
    
    # 位势高度等值线
    if level == 500:
        levels_z = np.arange(5400, 5960, 40)  # 500hPa
        levels_fill = np.arange(5400, 5960, 20)
    else:  # 850hPa
        levels_z = np.arange(1320, 1600, 20)
        levels_fill = np.arange(1320, 1600, 10)
    
    # 填充等值线
    cf = ax.contourf(lon2d, lat2d, z.values, levels=levels_fill, 
                     cmap='RdYlBu_r', extend='both',
                     transform=ccrs.PlateCarree())
    
    # 等值线
    cs = ax.contour(lon2d, lat2d, z.values, levels=levels_z[::2],
                    colors='black', linewidths=0.8,
                    transform=ccrs.PlateCarree())
    ax.clabel(cs, inline=True, fontsize=8, fmt='%d')
    
    # 风场 (稀疏显示)
    skip = 4
    wind_speed = calculate_wind_speed(u.values, v.values)
    q = ax.quiver(lon2d[::skip, ::skip], lat2d[::skip, ::skip],
                  u.values[::skip, ::skip], v.values[::skip, ::skip],
                  wind_speed[::skip, ::skip],
                  cmap='YlOrRd', scale=300, width=0.003,
                  transform=ccrs.PlateCarree())
    
    # 标记郑州位置
    ax.plot(center_lon, center_lat, 'r*', markersize=15, 
            transform=ccrs.PlateCarree(), label='郑州')
    
    # 设置经纬度刻度
    ax.set_xticks(np.arange(region[0], region[1]+1, 10), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(region[2], region[3]+1, 5), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5, linestyle='--')
    
    return cf, q


def plot_moisture_flux(q, u, v, ax, region, level=850):
    """
    绘制水汽通量
    
    参数:
        q: 比湿 (kg/kg)
        u, v: 风速分量 (m/s)
        ax: matplotlib轴对象
        region: 绘图区域
        level: 气压层 (hPa)
    """
    lon = q.longitude.values if 'longitude' in q.dims else q.lon.values
    lat = q.latitude.values if 'latitude' in q.dims else q.lat.values
    
    # 计算水汽通量 (kg/m/s)
    g = 9.80665
    p = level * 100  # hPa -> Pa
    qu = q * u * p / g
    qv = q * v * p / g
    
    # 水汽通量大小
    q_flux = np.sqrt(qu**2 + qv**2)
    
    ax.set_extent(region, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    
    lon2d, lat2d = np.meshgrid(lon, lat)
    
    # 填充水汽通量大小
    levels = np.arange(0, 350, 25)
    cf = ax.contourf(lon2d, lat2d, q_flux.values * 1000,  # 转换单位
                     levels=levels, cmap='Blues', extend='max',
                     transform=ccrs.PlateCarree())
    
    # 水汽通量矢量
    skip = 4
    ax.quiver(lon2d[::skip, ::skip], lat2d[::skip, ::skip],
              qu.values[::skip, ::skip] * 1000, 
              qv.values[::skip, ::skip] * 1000,
              scale=5000, width=0.003, color='green',
              transform=ccrs.PlateCarree())
    
    # 标记郑州
    ax.plot(ZHENGZHOU_LON, ZHENGZHOU_LAT, 'r*', markersize=15,
            transform=ccrs.PlateCarree())
    
    ax.set_xticks(np.arange(region[0], region[1]+1, 10), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(region[2], region[3]+1, 5), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5, linestyle='--')
    
    return cf


def calculate_moisture_convergence(q, u, v, lon, lat):
    """
    计算水汽通量散度
    
    参数:
        q: 比湿
        u, v: 风速分量
        lon, lat: 经纬度
    
    返回:
        水汽通量散度
    """
    # 转换为numpy数组
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    
    # 地球半径 (m)
    R = 6371000
    
    # 计算水汽通量
    qu = q * u
    qv = q * v
    
    # 计算网格间距
    dlat = np.abs(lat[1] - lat[0])
    dlon = np.abs(lon[1] - lon[0])
    
    # 计算散度
    dqu_dlon = np.gradient(qu, np.deg2rad(dlon), axis=-1)
    dqv_dlat = np.gradient(qv, np.deg2rad(dlat), axis=-2)
    
    lat2d = np.broadcast_to(lat_rad[:, np.newaxis], qu.shape) if qu.ndim == 2 else lat_rad
    
    # 散度计算
    div = (1 / (R * np.cos(lat2d))) * dqu_dlon + (1 / R) * dqv_dlat
    
    return -div * 1e6  # 转换单位，辐合为正


def main():
    """主程序"""
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    print("=" * 60)
    print("郑州极端降水环流分析")
    print(f"分析时间: {ANALYSIS_DATE}")
    print("=" * 60)
    
    # 加载数据
    print("\n加载ERA5数据...")
    data = load_era5_data(DATA_DIR, ANALYSIS_DATE)
    
    if data is None:
        print("数据加载失败，使用示例数据进行演示...")
        # 创建示例数据用于演示
        lon = np.arange(ANALYSIS_REGION[0], ANALYSIS_REGION[1] + 0.25, 0.25)
        lat = np.arange(ANALYSIS_REGION[3], ANALYSIS_REGION[2] - 0.25, -0.25)
        
        # 创建示例位势高度场 (500hPa)
        lon2d, lat2d = np.meshgrid(lon, lat)
        z500 = 5700 + 100 * np.sin(np.deg2rad(lon2d - 110)) * np.cos(np.deg2rad(lat2d - 35))
        z850 = 1450 + 50 * np.sin(np.deg2rad(lon2d - 115)) * np.cos(np.deg2rad(lat2d - 30))
        
        # 创建示例风场
        u500 = 10 + 5 * np.sin(np.deg2rad(lat2d))
        v500 = 2 * np.cos(np.deg2rad(lon2d - 110))
        u850 = 8 + 3 * np.sin(np.deg2rad(lat2d))
        v850 = 5 * np.cos(np.deg2rad(lon2d - 115))
        
        # 创建示例比湿
        q850 = 0.015 * np.exp(-((lon2d - 115)**2 + (lat2d - 30)**2) / 200)
        
        # 转换为DataArray
        z500 = xr.DataArray(z500, dims=['latitude', 'longitude'],
                           coords={'latitude': lat, 'longitude': lon})
        z850 = xr.DataArray(z850, dims=['latitude', 'longitude'],
                           coords={'latitude': lat, 'longitude': lon})
        u500 = xr.DataArray(u500, dims=['latitude', 'longitude'],
                           coords={'latitude': lat, 'longitude': lon})
        v500 = xr.DataArray(v500, dims=['latitude', 'longitude'],
                           coords={'latitude': lat, 'longitude': lon})
        u850 = xr.DataArray(u850, dims=['latitude', 'longitude'],
                           coords={'latitude': lat, 'longitude': lon})
        v850 = xr.DataArray(v850, dims=['latitude', 'longitude'],
                           coords={'latitude': lat, 'longitude': lon})
        q850 = xr.DataArray(q850, dims=['latitude', 'longitude'],
                           coords={'latitude': lat, 'longitude': lon})
    else:
        # 从实际数据中提取
        pl_data = data['pl']
        
        # 选择时间和气压层
        if 'time' in pl_data.dims:
            pl_data = pl_data.sel(time=ANALYSIS_DATE, method='nearest')
        
        # 获取变量
        if 'level' in pl_data.dims:
            pl_data = pl_data.rename({'level': 'pressure_level'})
        
        z500 = pl_data['z'].sel(pressure_level=500) / 9.80665
        z850 = pl_data['z'].sel(pressure_level=850) / 9.80665
        u500 = pl_data['u'].sel(pressure_level=500)
        v500 = pl_data['v'].sel(pressure_level=500)
        u850 = pl_data['u'].sel(pressure_level=850)
        v850 = pl_data['v'].sel(pressure_level=850)
        q850 = pl_data['q'].sel(pressure_level=850)
    
    # ==================== 绘图 ====================
    print("\n绘制环流分析图...")
    
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(f'郑州"7·20"极端暴雨环流分析\n{ANALYSIS_DATE.strftime("%Y年%m月%d日 %H时")} (UTC)',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 子图1: 500hPa位势高度和风场
    ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
    cf1, q1 = plot_geopotential_wind(z500, u500, v500, 500, ax1, ANALYSIS_REGION)
    ax1.set_title('a. 500hPa位势高度场和风场', loc='left', fontsize=12, fontweight='bold')
    cbar1 = plt.colorbar(cf1, ax=ax1, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar1.set_label('位势高度 (gpm)', fontsize=10)
    
    # 子图2: 850hPa位势高度和风场
    ax2 = fig.add_subplot(2, 2, 2, projection=ccrs.PlateCarree())
    cf2, q2 = plot_geopotential_wind(z850, u850, v850, 850, ax2, ANALYSIS_REGION)
    ax2.set_title('b. 850hPa位势高度场和风场', loc='left', fontsize=12, fontweight='bold')
    cbar2 = plt.colorbar(cf2, ax=ax2, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar2.set_label('位势高度 (gpm)', fontsize=10)
    
    # 子图3: 850hPa水汽通量
    ax3 = fig.add_subplot(2, 2, 3, projection=ccrs.PlateCarree())
    cf3 = plot_moisture_flux(q850, u850, v850, ax3, ANALYSIS_REGION, level=850)
    ax3.set_title('c. 850hPa水汽通量', loc='left', fontsize=12, fontweight='bold')
    cbar3 = plt.colorbar(cf3, ax=ax3, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar3.set_label('水汽通量 (g/(m·s))', fontsize=10)
    
    # 子图4: 水汽通量辐合
    ax4 = fig.add_subplot(2, 2, 4, projection=ccrs.PlateCarree())
    ax4.set_extent(ANALYSIS_REGION, crs=ccrs.PlateCarree())
    ax4.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax4.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    
    # 计算水汽通量辐合
    lon = q850.longitude.values if 'longitude' in q850.dims else q850.lon.values
    lat = q850.latitude.values if 'latitude' in q850.dims else q850.lat.values
    q_conv = calculate_moisture_convergence(q850.values, u850.values, v850.values, lon, lat)
    
    lon2d, lat2d = np.meshgrid(lon, lat)
    levels_conv = np.arange(-5, 5.5, 0.5)
    cf4 = ax4.contourf(lon2d, lat2d, q_conv, levels=levels_conv,
                       cmap='BrBG', extend='both',
                       transform=ccrs.PlateCarree())
    ax4.plot(ZHENGZHOU_LON, ZHENGZHOU_LAT, 'r*', markersize=15,
             transform=ccrs.PlateCarree())
    ax4.set_xticks(np.arange(ANALYSIS_REGION[0], ANALYSIS_REGION[1]+1, 10), crs=ccrs.PlateCarree())
    ax4.set_yticks(np.arange(ANALYSIS_REGION[2], ANALYSIS_REGION[3]+1, 5), crs=ccrs.PlateCarree())
    ax4.xaxis.set_major_formatter(LongitudeFormatter())
    ax4.yaxis.set_major_formatter(LatitudeFormatter())
    ax4.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5, linestyle='--')
    ax4.set_title('d. 850hPa水汽通量辐合', loc='left', fontsize=12, fontweight='bold')
    cbar4 = plt.colorbar(cf4, ax=ax4, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar4.set_label('水汽通量辐合 (×10⁻⁶ g/(kg·s))', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(OUTPUT_DIR, 'zhengzhou_circulation_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n图像已保存: {output_path}")
    
    plt.show()
    
    print("\n分析完成!")


if __name__ == '__main__':
    main()
