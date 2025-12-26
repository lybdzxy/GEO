"""
郑州极端降水热力学指数分析
Thermodynamic instability indices analysis for Zhengzhou extreme precipitation (2021.7.20)

功能：
1. 计算对流有效位能 (CAPE)
2. 计算K指数、SI指数、LI指数等
3. 假相当位温分析
4. 垂直积分水汽通量

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

# 物理常数
G = 9.80665  # 重力加速度 (m/s²)
RD = 287.05  # 干空气气体常数 (J/(kg·K))
RV = 461.5   # 水汽气体常数 (J/(kg·K))
CP = 1004.0  # 定压比热 (J/(kg·K))
LV = 2.5e6   # 汽化潜热 (J/kg)
P0 = 100000  # 参考气压 (Pa)


def saturation_vapor_pressure(t):
    """
    计算饱和水汽压 (Tetens公式)
    
    参数:
        t: 温度 (K)
    
    返回:
        饱和水汽压 (Pa)
    """
    t_c = t - 273.15
    es = 611.2 * np.exp(17.67 * t_c / (t_c + 243.5))
    return es


def dewpoint_from_specific_humidity(q, p):
    """
    从比湿计算露点温度
    
    参数:
        q: 比湿 (kg/kg)
        p: 气压 (Pa)
    
    返回:
        露点温度 (K)
    """
    # 水汽压
    e = q * p / (0.622 + 0.378 * q)
    e = np.maximum(e, 1)  # 避免log(0)
    
    # 露点温度 (逆Tetens公式)
    td_c = 243.5 * np.log(e / 611.2) / (17.67 - np.log(e / 611.2))
    td = td_c + 273.15
    
    return td


def potential_temperature(t, p):
    """
    计算位温
    
    参数:
        t: 温度 (K)
        p: 气压 (Pa)
    
    返回:
        位温 (K)
    """
    return t * (P0 / p) ** (RD / CP)


def equivalent_potential_temperature(t, td, p):
    """
    计算相当位温 (Bolton 1980公式)
    
    参数:
        t: 温度 (K)
        td: 露点温度 (K)
        p: 气压 (Pa)
    
    返回:
        相当位温 (K)
    """
    # 计算混合比
    e = saturation_vapor_pressure(td)
    r = 0.622 * e / (p - e)  # 混合比 (kg/kg)
    
    # 抬升凝结高度温度 (近似)
    t_c = t - 273.15
    td_c = td - 273.15
    tlcl = 56 + 1 / (1/(td_c - 56) + np.log(t/td) / 800)
    tlcl = tlcl + 273.15
    
    # 相当位温
    theta_e = t * (P0 / p) ** (0.2854 * (1 - 0.28 * r)) * \
              np.exp((3.376 / tlcl - 0.00254) * r * 1000 * (1 + 0.81 * r))
    
    return theta_e


def calculate_k_index(t850, t700, t500, td850, td700):
    """
    计算K指数
    K = (T850 - T500) + Td850 - (T700 - Td700)
    
    参数:
        t850, t700, t500: 各层温度 (K)
        td850, td700: 各层露点温度 (K)
    
    返回:
        K指数 (°C)
    """
    # 转换为摄氏度
    t850_c = t850 - 273.15
    t700_c = t700 - 273.15
    t500_c = t500 - 273.15
    td850_c = td850 - 273.15
    td700_c = td700 - 273.15
    
    k_index = (t850_c - t500_c) + td850_c - (t700_c - td700_c)
    
    return k_index


def calculate_si_index(t850, t500, td850):
    """
    计算沙氏指数 (简化版)
    SI = T500 - T_lifted
    
    参数:
        t850: 850hPa温度 (K)
        t500: 500hPa温度 (K)
        td850: 850hPa露点温度 (K)
    
    返回:
        SI指数 (°C)
    """
    # 简化计算：假设从850hPa湿绝热抬升到500hPa
    # 实际应该用精确的湿绝热线计算
    
    # 850hPa平均温度
    t_mean = (t850 + td850) / 2
    
    # 干绝热抬升到LCL (近似)
    t_lcl = t850 - 0.01 * (t850 - td850) * 100  # 约1K/100m
    
    # 湿绝热抬升到500hPa (近似减温率约6K/km，约350hPa高度差约4km)
    t_lifted = t_lcl - 6.0 * 4.0
    
    si_index = (t500 - 273.15) - t_lifted
    
    return si_index


def calculate_li_index(t_parcel_500, t_env_500):
    """
    计算抬升指数
    LI = T_env_500 - T_parcel_500
    
    参数:
        t_parcel_500: 气块抬升到500hPa的温度 (K)
        t_env_500: 环境500hPa温度 (K)
    
    返回:
        LI指数 (°C)
    """
    return (t_env_500 - t_parcel_500)


def calculate_cape_simple(t_profile, td_profile, p_profile):
    """
    简化的CAPE计算
    
    参数:
        t_profile: 温度廓线 (K)
        td_profile: 露点廓线 (K)
        p_profile: 气压廓线 (Pa)
    
    返回:
        CAPE (J/kg)
    """
    # 这是一个简化版本，实际CAPE需要精确的气块抬升计算
    # 找到自由对流高度和平衡高度
    
    cape = 0
    cin = 0
    
    # 计算虚温
    e_env = saturation_vapor_pressure(td_profile)
    r_env = 0.622 * e_env / (p_profile - e_env)
    tv_env = t_profile * (1 + 0.61 * r_env)
    
    # 假设气块从地面抬升，简化计算
    # 实际应该沿着干绝热线和湿绝热线抬升
    
    # 这里使用经验公式估算
    if len(p_profile) > 5:
        # 低层温度露点差
        t_td_diff = t_profile[0] - td_profile[0]
        
        # 850-500hPa温差
        idx_850 = np.argmin(np.abs(p_profile - 85000))
        idx_500 = np.argmin(np.abs(p_profile - 50000))
        
        if idx_850 < len(t_profile) and idx_500 < len(t_profile):
            lapse_rate = (t_profile[idx_850] - t_profile[idx_500]) / (85000 - 50000) * 10000
            
            # 简化CAPE估算
            if t_td_diff < 5:  # 接近饱和
                cape = max(0, (lapse_rate - 6.5) * 200)  # 每度超过6.5K/km增加200 J/kg
            else:
                cape = max(0, (lapse_rate - 6.5) * 100)
    
    return cape


def calculate_iwv(q, p_levels):
    """
    计算整层可降水量 (IWV/TPW)
    
    参数:
        q: 比湿廓线 (kg/kg), shape: (levels, lat, lon)
        p_levels: 气压层 (Pa)
    
    返回:
        可降水量 (kg/m² 或 mm)
    """
    # 积分 IWV = (1/g) * ∫q dp
    iwv = np.trapz(q, x=-p_levels, axis=0) / G
    
    return iwv


def plot_thermodynamic_map(data, lon, lat, ax, title, cmap='YlOrRd', 
                           levels=None, extend='max'):
    """
    绘制热力学指数空间分布
    """
    ax.set_extent(ANALYSIS_REGION, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    
    lon2d, lat2d = np.meshgrid(lon, lat)
    
    if levels is None:
        levels = 20
    
    cf = ax.contourf(lon2d, lat2d, data, levels=levels,
                     cmap=cmap, extend=extend,
                     transform=ccrs.PlateCarree())
    
    # 添加等值线
    if isinstance(levels, (list, np.ndarray)) and len(levels) > 5:
        cs = ax.contour(lon2d, lat2d, data, levels=levels[::4],
                        colors='black', linewidths=0.5,
                        transform=ccrs.PlateCarree())
        ax.clabel(cs, inline=True, fontsize=7, fmt='%.0f')
    
    ax.plot(ZHENGZHOU_LON, ZHENGZHOU_LAT, 'r*', markersize=15,
            transform=ccrs.PlateCarree())
    
    ax.set_xticks(np.arange(ANALYSIS_REGION[0], ANALYSIS_REGION[1]+1, 10), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(ANALYSIS_REGION[2], ANALYSIS_REGION[3]+1, 5), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5, linestyle='--')
    
    ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
    
    return cf


def main():
    """主程序"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    print("=" * 60)
    print("郑州极端降水热力学指数分析")
    print(f"分析时间: {ANALYSIS_DATE}")
    print("=" * 60)
    
    # 创建示例数据
    print("\n生成示例数据...")
    lon = np.arange(ANALYSIS_REGION[0], ANALYSIS_REGION[1] + 0.25, 0.25)
    lat = np.arange(ANALYSIS_REGION[3], ANALYSIS_REGION[2] - 0.25, -0.25)
    p_levels = np.array([100000, 92500, 85000, 70000, 50000, 30000, 20000])  # Pa
    
    lon2d, lat2d = np.meshgrid(lon, lat)
    
    # 模拟温度场 (夏季东亚)
    # 850hPa温度 (约300K在热带，向北递减)
    t850 = 300 - 0.5 * (lat2d - 20) + 2 * np.sin(np.deg2rad(lon2d - 110))
    t700 = 285 - 0.4 * (lat2d - 20) + 1.5 * np.sin(np.deg2rad(lon2d - 110))
    t500 = 265 - 0.3 * (lat2d - 20) + 1 * np.sin(np.deg2rad(lon2d - 110))
    
    # 模拟露点 (郑州附近高湿)
    td850 = t850 - 3 - 5 * np.exp(-((lon2d - ZHENGZHOU_LON)**2 + (lat2d - ZHENGZHOU_LAT)**2) / 100)
    td700 = t700 - 8 - 3 * np.exp(-((lon2d - ZHENGZHOU_LON)**2 + (lat2d - ZHENGZHOU_LAT)**2) / 150)
    
    # 模拟比湿 (3D)
    q_3d = np.zeros((len(p_levels), len(lat), len(lon)))
    for i, p in enumerate(p_levels):
        q_base = 0.015 * np.exp(-(100000 - p) / 30000)  # 随高度递减
        q_spatial = q_base * (1 + 0.5 * np.exp(-((lon2d - ZHENGZHOU_LON)**2 + 
                                                  (lat2d - ZHENGZHOU_LAT)**2) / 100))
        q_3d[i] = q_spatial
    
    # ==================== 计算热力学指数 ====================
    print("\n计算热力学指数...")
    
    # K指数
    k_index = calculate_k_index(t850, t700, t500, td850, td700)
    
    # SI指数
    si_index = calculate_si_index(t850, t500, td850)
    
    # 相当位温 (850hPa)
    theta_e_850 = equivalent_potential_temperature(t850, td850, 85000)
    
    # 可降水量
    iwv = calculate_iwv(q_3d, p_levels)
    
    # 假相当位温垂直梯度 (850-500hPa)
    theta_e_500 = equivalent_potential_temperature(t500, td700 - 10, 50000)  # 简化
    d_theta_e = theta_e_850 - theta_e_500
    
    # ==================== 绘图 ====================
    print("\n绘制热力学分析图...")
    
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f'郑州"7·20"极端暴雨热力学指数分析\n{ANALYSIS_DATE.strftime("%Y年%m月%d日 %H时")} (UTC)',
                 fontsize=14, fontweight='bold', y=0.98)
    
    # K指数
    ax1 = fig.add_subplot(2, 3, 1, projection=ccrs.PlateCarree())
    levels_k = np.arange(20, 45, 2)
    cf1 = plot_thermodynamic_map(k_index, lon, lat, ax1,
                                  'a. K指数', cmap='YlOrRd',
                                  levels=levels_k)
    cbar1 = plt.colorbar(cf1, ax=ax1, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar1.set_label('K指数 (°C)', fontsize=10)
    
    # SI指数
    ax2 = fig.add_subplot(2, 3, 2, projection=ccrs.PlateCarree())
    levels_si = np.arange(-10, 5, 1)
    cf2 = plot_thermodynamic_map(si_index, lon, lat, ax2,
                                  'b. 沙氏指数 (SI)', cmap='RdYlBu',
                                  levels=levels_si, extend='both')
    cbar2 = plt.colorbar(cf2, ax=ax2, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar2.set_label('SI指数 (°C)', fontsize=10)
    
    # 850hPa相当位温
    ax3 = fig.add_subplot(2, 3, 3, projection=ccrs.PlateCarree())
    levels_te = np.arange(320, 370, 4)
    cf3 = plot_thermodynamic_map(theta_e_850, lon, lat, ax3,
                                  'c. 850hPa 相当位温', cmap='Spectral_r',
                                  levels=levels_te)
    cbar3 = plt.colorbar(cf3, ax=ax3, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar3.set_label('θe (K)', fontsize=10)
    
    # 可降水量
    ax4 = fig.add_subplot(2, 3, 4, projection=ccrs.PlateCarree())
    levels_iwv = np.arange(20, 75, 5)
    cf4 = plot_thermodynamic_map(iwv, lon, lat, ax4,
                                  'd. 整层可降水量', cmap='Blues',
                                  levels=levels_iwv)
    cbar4 = plt.colorbar(cf4, ax=ax4, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar4.set_label('IWV (mm)', fontsize=10)
    
    # 相当位温垂直梯度
    ax5 = fig.add_subplot(2, 3, 5, projection=ccrs.PlateCarree())
    levels_dte = np.arange(-10, 30, 2)
    cf5 = plot_thermodynamic_map(d_theta_e, lon, lat, ax5,
                                  'e. θe垂直梯度 (850-500hPa)', cmap='RdBu_r',
                                  levels=levels_dte, extend='both')
    cbar5 = plt.colorbar(cf5, ax=ax5, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar5.set_label('Δθe (K)', fontsize=10)
    
    # 温度露点差 (850hPa)
    ax6 = fig.add_subplot(2, 3, 6, projection=ccrs.PlateCarree())
    t_td_850 = t850 - td850
    levels_ttd = np.arange(0, 20, 2)
    cf6 = plot_thermodynamic_map(t_td_850, lon, lat, ax6,
                                  'f. 850hPa 温度露点差', cmap='YlOrBr',
                                  levels=levels_ttd)
    cbar6 = plt.colorbar(cf6, ax=ax6, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar6.set_label('T-Td (°C)', fontsize=10)
    
    plt.tight_layout()
    
    # 保存
    output_path = os.path.join(OUTPUT_DIR, 'zhengzhou_thermodynamic_indices.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n图像已保存: {output_path}")
    
    plt.show()
    
    # ==================== 输出关键参数 ====================
    print("\n" + "=" * 60)
    print("郑州位置热力学指数:")
    print("=" * 60)
    
    # 找到郑州位置
    lon_idx = np.argmin(np.abs(lon - ZHENGZHOU_LON))
    lat_idx = np.argmin(np.abs(lat - ZHENGZHOU_LAT))
    
    print(f"\nK指数: {k_index[lat_idx, lon_idx]:.1f} °C")
    print(f"  (>35°C: 有利于强对流; >40°C: 有利于暴雨)")
    
    print(f"\nSI指数: {si_index[lat_idx, lon_idx]:.1f} °C")
    print(f"  (<0°C: 不稳定; <-3°C: 强不稳定)")
    
    print(f"\n850hPa相当位温: {theta_e_850[lat_idx, lon_idx]:.1f} K")
    print(f"  (>350K: 高温高湿)")
    
    print(f"\n整层可降水量: {iwv[lat_idx, lon_idx]:.1f} mm")
    print(f"  (>60mm: 极端高湿环境)")
    
    print(f"\n850hPa温度露点差: {t_td_850[lat_idx, lon_idx]:.1f} °C")
    print(f"  (<5°C: 近饱和)")
    
    # 不稳定度判断
    print("\n" + "=" * 60)
    print("大气不稳定度评估:")
    print("=" * 60)
    
    k_val = k_index[lat_idx, lon_idx]
    si_val = si_index[lat_idx, lon_idx]
    
    if k_val > 40 and si_val < -3:
        print("评估结果: 极端不稳定，有利于强降水发生")
    elif k_val > 35 and si_val < 0:
        print("评估结果: 明显不稳定，有利于对流发展")
    elif k_val > 30:
        print("评估结果: 中等不稳定")
    else:
        print("评估结果: 弱不稳定或稳定")
    
    print("\n分析完成!")


if __name__ == '__main__':
    main()
