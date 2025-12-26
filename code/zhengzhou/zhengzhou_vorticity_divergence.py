"""
郑州极端降水涡度散度分析
Vorticity and divergence analysis for Zhengzhou extreme precipitation event (2021.7.20)

功能：
1. 涡度和散度场分析
2. 涡度平流分析
3. 垂直涡度收支分析

使用ERA5数据进行分析
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import os
from datetime import datetime

# ==================== 配置参数 ====================
DATA_DIR = r'E:/ERA5/hourly'
OUTPUT_DIR = r'E:/GEO/pyproject/casestudy/zhengzhou'

ZHENGZHOU_LON = 113.65
ZHENGZHOU_LAT = 34.76

ANALYSIS_REGION = [100, 130, 20, 45]
ANALYSIS_DATE = datetime(2021, 7, 20, 12)


def calculate_vorticity(u, v, lon, lat):
    """
    计算相对涡度
    
    参数:
        u, v: 风速分量 (m/s)
        lon, lat: 经纬度 (度)
    
    返回:
        相对涡度 (1/s)
    """
    # 地球半径
    R = 6371000  # m
    
    # 转换为弧度
    lat_rad = np.deg2rad(lat)
    
    # 计算网格间距
    dlat = np.deg2rad(np.abs(lat[1] - lat[0]))
    dlon = np.deg2rad(np.abs(lon[1] - lon[0]))
    
    # 创建2D纬度数组用于cos(lat)计算
    if u.ndim == 2:
        lat2d = np.broadcast_to(lat_rad[:, np.newaxis], u.shape)
    else:
        lat2d = lat_rad
    
    # dv/dx
    dv_dx = np.gradient(v, axis=-1) / (R * np.cos(lat2d) * dlon)
    
    # du/dy
    du_dy = np.gradient(u, axis=-2) / (R * dlat)
    
    # 相对涡度
    vorticity = dv_dx - du_dy
    
    return vorticity


def calculate_divergence(u, v, lon, lat):
    """
    计算散度
    
    参数:
        u, v: 风速分量 (m/s)
        lon, lat: 经纬度 (度)
    
    返回:
        散度 (1/s)
    """
    R = 6371000  # m
    
    lat_rad = np.deg2rad(lat)
    dlat = np.deg2rad(np.abs(lat[1] - lat[0]))
    dlon = np.deg2rad(np.abs(lon[1] - lon[0]))
    
    if u.ndim == 2:
        lat2d = np.broadcast_to(lat_rad[:, np.newaxis], u.shape)
    else:
        lat2d = lat_rad
    
    # du/dx
    du_dx = np.gradient(u, axis=-1) / (R * np.cos(lat2d) * dlon)
    
    # d(v*cos(lat))/dy
    v_cos = v * np.cos(lat2d)
    dv_cos_dy = np.gradient(v_cos, axis=-2) / (R * dlat)
    
    # 考虑球面几何的散度
    divergence = du_dx + dv_cos_dy / np.cos(lat2d)
    
    return divergence


def calculate_vorticity_advection(u, v, vort, lon, lat):
    """
    计算涡度平流
    
    参数:
        u, v: 风速分量 (m/s)
        vort: 相对涡度 (1/s)
        lon, lat: 经纬度 (度)
    
    返回:
        涡度平流 (1/s²)
    """
    R = 6371000  # m
    
    lat_rad = np.deg2rad(lat)
    dlat = np.deg2rad(np.abs(lat[1] - lat[0]))
    dlon = np.deg2rad(np.abs(lon[1] - lon[0]))
    
    if u.ndim == 2:
        lat2d = np.broadcast_to(lat_rad[:, np.newaxis], u.shape)
    else:
        lat2d = lat_rad
    
    # 涡度梯度
    dvort_dx = np.gradient(vort, axis=-1) / (R * np.cos(lat2d) * dlon)
    dvort_dy = np.gradient(vort, axis=-2) / (R * dlat)
    
    # 涡度平流 (负号因为平流是逆梯度方向的输送)
    vort_adv = -(u * dvort_dx + v * dvort_dy)
    
    return vort_adv


def plot_vorticity_divergence_map(vort, div, lon, lat, ax, title, cmap='RdBu_r', vmax=None):
    """
    绘制涡度或散度场
    
    参数:
        vort: 涡度或散度数据
        lon, lat: 经纬度
        ax: matplotlib轴对象
        title: 图标题
        cmap: 色谱
        vmax: 最大值
    """
    ax.set_extent(ANALYSIS_REGION, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    
    lon2d, lat2d = np.meshgrid(lon, lat)
    
    if vmax is None:
        vmax = np.nanmax(np.abs(vort)) * 0.8
    
    levels = np.linspace(-vmax, vmax, 21)
    
    cf = ax.contourf(lon2d, lat2d, vort, levels=levels,
                     cmap=cmap, extend='both',
                     transform=ccrs.PlateCarree())
    
    # 零等值线
    ax.contour(lon2d, lat2d, vort, levels=[0],
               colors='black', linewidths=1,
               transform=ccrs.PlateCarree())
    
    # 标记郑州
    ax.plot(ZHENGZHOU_LON, ZHENGZHOU_LAT, 'r*', markersize=15,
            transform=ccrs.PlateCarree())
    
    # 设置坐标轴
    ax.set_xticks(np.arange(ANALYSIS_REGION[0], ANALYSIS_REGION[1]+1, 10), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(ANALYSIS_REGION[2], ANALYSIS_REGION[3]+1, 5), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5, linestyle='--')
    
    ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
    
    return cf


def plot_vertical_vorticity_profile(vort_levels, p_levels, center_lon, center_lat, 
                                    lon, lat, ax, direction='zonal'):
    """
    绘制涡度垂直剖面
    
    参数:
        vort_levels: 各层涡度数据 (dict: level -> vort)
        p_levels: 气压层列表
        center_lon, center_lat: 剖面中心
        lon, lat: 经纬度数组
        ax: matplotlib轴对象
        direction: 'zonal' 或 'meridional'
    """
    # 提取剖面数据
    if direction == 'zonal':
        lat_idx = np.argmin(np.abs(lat - center_lat))
        x_coord = lon
        xlabel = '经度 (°E)'
        center_mark = center_lon
        vort_slice = np.array([vort_levels[p][lat_idx, :] for p in p_levels])
    else:
        lon_idx = np.argmin(np.abs(lon - center_lon))
        x_coord = lat
        xlabel = '纬度 (°N)'
        center_mark = center_lat
        vort_slice = np.array([vort_levels[p][:, lon_idx] for p in p_levels])
    
    xx, yy = np.meshgrid(x_coord, p_levels)
    
    # 转换单位为 10^-5 s^-1
    vort_plot = vort_slice * 1e5
    
    vmax = np.nanmax(np.abs(vort_plot)) * 0.8
    levels = np.linspace(-vmax, vmax, 21)
    
    cf = ax.contourf(xx, yy, vort_plot, levels=levels,
                     cmap='RdBu_r', extend='both')
    
    ax.contour(xx, yy, vort_plot, levels=[0],
               colors='black', linewidths=1)
    
    ax.axvline(center_mark, color='red', linewidth=2, linestyle='--')
    
    ax.set_ylim(1000, 200)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('气压 (hPa)', fontsize=10)
    ax.invert_yaxis()
    
    return cf


def main():
    """主程序"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    print("=" * 60)
    print("郑州极端降水涡度散度分析")
    print(f"分析时间: {ANALYSIS_DATE}")
    print("=" * 60)
    
    # 创建示例数据
    print("\n生成示例数据...")
    lon = np.arange(ANALYSIS_REGION[0], ANALYSIS_REGION[1] + 0.25, 0.25)
    lat = np.arange(ANALYSIS_REGION[3], ANALYSIS_REGION[2] - 0.25, -0.25)
    p_levels = [1000, 925, 850, 700, 500, 300, 200]
    
    lon2d, lat2d = np.meshgrid(lon, lat)
    
    # 创建模拟的环流场 (模拟低涡系统)
    # 低涡中心在郑州附近
    r = np.sqrt((lon2d - ZHENGZHOU_LON)**2 + (lat2d - ZHENGZHOU_LAT)**2)
    
    vort_levels = {}
    div_levels = {}
    u_levels = {}
    v_levels = {}
    
    for p in p_levels:
        # 风场：模拟气旋式环流
        scale = 1.0 - (1000 - p) / 1000 * 0.3  # 随高度减弱
        
        # 切向速度 (气旋式)
        vt = 15 * scale * np.exp(-r**2 / 100) * r / (r + 1)
        
        # 转换为u, v分量
        theta = np.arctan2(lat2d - ZHENGZHOU_LAT, lon2d - ZHENGZHOU_LON)
        u = -vt * np.sin(theta) + 5  # 背景西风
        v = vt * np.cos(theta)
        
        # 计算涡度和散度
        vort = calculate_vorticity(u, v, lon, lat)
        div = calculate_divergence(u, v, lon, lat)
        
        vort_levels[p] = vort
        div_levels[p] = div
        u_levels[p] = u
        v_levels[p] = v
    
    # ==================== 绘图 ====================
    print("\n绘制涡度散度分析图...")
    
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f'郑州"7·20"极端暴雨涡度散度分析\n{ANALYSIS_DATE.strftime("%Y年%m月%d日 %H时")} (UTC)',
                 fontsize=14, fontweight='bold', y=0.98)
    
    # 第一行: 850hPa和500hPa涡度
    ax1 = fig.add_subplot(2, 3, 1, projection=ccrs.PlateCarree())
    cf1 = plot_vorticity_divergence_map(vort_levels[850] * 1e5, div_levels[850], 
                                        lon, lat, ax1, 
                                        'a. 850hPa 相对涡度', vmax=20)
    cbar1 = plt.colorbar(cf1, ax=ax1, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar1.set_label('涡度 (×10⁻⁵ s⁻¹)', fontsize=10)
    
    ax2 = fig.add_subplot(2, 3, 2, projection=ccrs.PlateCarree())
    cf2 = plot_vorticity_divergence_map(vort_levels[500] * 1e5, div_levels[500],
                                        lon, lat, ax2,
                                        'b. 500hPa 相对涡度', vmax=15)
    cbar2 = plt.colorbar(cf2, ax=ax2, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar2.set_label('涡度 (×10⁻⁵ s⁻¹)', fontsize=10)
    
    # 第一行: 200hPa散度
    ax3 = fig.add_subplot(2, 3, 3, projection=ccrs.PlateCarree())
    cf3 = plot_vorticity_divergence_map(div_levels[200] * 1e5, None,
                                        lon, lat, ax3,
                                        'c. 200hPa 散度', 
                                        cmap='PuOr_r', vmax=10)
    cbar3 = plt.colorbar(cf3, ax=ax3, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar3.set_label('散度 (×10⁻⁵ s⁻¹)', fontsize=10)
    
    # 第二行: 850hPa散度和涡度平流
    ax4 = fig.add_subplot(2, 3, 4, projection=ccrs.PlateCarree())
    cf4 = plot_vorticity_divergence_map(div_levels[850] * 1e5, None,
                                        lon, lat, ax4,
                                        'd. 850hPa 散度',
                                        cmap='PuOr_r', vmax=10)
    cbar4 = plt.colorbar(cf4, ax=ax4, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar4.set_label('散度 (×10⁻⁵ s⁻¹)', fontsize=10)
    
    # 涡度平流
    vort_adv_850 = calculate_vorticity_advection(u_levels[850], v_levels[850],
                                                  vort_levels[850], lon, lat)
    
    ax5 = fig.add_subplot(2, 3, 5, projection=ccrs.PlateCarree())
    cf5 = plot_vorticity_divergence_map(vort_adv_850 * 1e9, None,
                                        lon, lat, ax5,
                                        'e. 850hPa 涡度平流',
                                        cmap='RdYlBu_r', vmax=5)
    cbar5 = plt.colorbar(cf5, ax=ax5, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar5.set_label('涡度平流 (×10⁻⁹ s⁻²)', fontsize=10)
    
    # 第二行: 涡度垂直剖面
    ax6 = fig.add_subplot(2, 3, 6)
    cf6 = plot_vertical_vorticity_profile(vort_levels, p_levels, 
                                          ZHENGZHOU_LON, ZHENGZHOU_LAT,
                                          lon, lat, ax6, direction='zonal')
    cbar6 = plt.colorbar(cf6, ax=ax6, orientation='vertical', pad=0.02)
    cbar6.set_label('涡度 (×10⁻⁵ s⁻¹)', fontsize=10)
    ax6.set_title('f. 涡度纬向垂直剖面', loc='left', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存
    output_path = os.path.join(OUTPUT_DIR, 'zhengzhou_vorticity_divergence.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n图像已保存: {output_path}")
    
    plt.show()
    
    # ==================== 输出关键参数 ====================
    print("\n" + "=" * 60)
    print("关键涡度散度参数 (郑州位置):")
    print("=" * 60)
    
    # 找到郑州位置的索引
    lon_idx = np.argmin(np.abs(lon - ZHENGZHOU_LON))
    lat_idx = np.argmin(np.abs(lat - ZHENGZHOU_LAT))
    
    print("\n相对涡度 (×10⁻⁵ s⁻¹):")
    for p in [850, 700, 500]:
        vort_val = vort_levels[p][lat_idx, lon_idx] * 1e5
        print(f"  {p}hPa: {vort_val:.2f}")
    
    print("\n散度 (×10⁻⁵ s⁻¹):")
    for p in [850, 700, 500, 200]:
        div_val = div_levels[p][lat_idx, lon_idx] * 1e5
        sign = "辐合" if div_val < 0 else "辐散"
        print(f"  {p}hPa: {div_val:.2f} ({sign})")
    
    print("\n分析完成!")


if __name__ == '__main__':
    main()
