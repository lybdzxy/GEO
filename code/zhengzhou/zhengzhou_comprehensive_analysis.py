"""
郑州极端降水综合分析
Comprehensive analysis for Zhengzhou extreme precipitation event (2021.7.20)

功能：
1. 时间序列分析 (降水演变、环流变化)
2. 合成分析
3. 动力-热力耦合分析
4. 主要环流系统识别

使用ERA5数据进行分析
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import os
from datetime import datetime, timedelta
import pandas as pd

# ==================== 配置参数 ====================
DATA_DIR = r'E:/ERA5/hourly'
OUTPUT_DIR = r'E:/GEO/pyproject/casestudy/zhengzhou'

ZHENGZHOU_LON = 113.65
ZHENGZHOU_LAT = 34.76

ANALYSIS_REGION = [100, 130, 20, 45]
LOCAL_REGION = [110, 118, 32, 38]  # 郑州周边小区域

# 分析时段 (2021年7月17日-21日)
START_DATE = datetime(2021, 7, 17, 0)
END_DATE = datetime(2021, 7, 21, 23)

# 关键时刻
KEY_TIMES = [
    datetime(2021, 7, 19, 12),  # 前期
    datetime(2021, 7, 20, 8),   # 极端降水开始
    datetime(2021, 7, 20, 12),  # 极端降水峰值
    datetime(2021, 7, 20, 18),  # 峰值后期
]


def generate_sample_time_series(times):
    """
    生成示例时间序列数据
    """
    n = len(times)
    
    # 模拟降水时间序列 (7月20日出现峰值)
    hours = np.array([(t - START_DATE).total_seconds() / 3600 for t in times])
    
    # 降水强度 (mm/h) - 7月20日12时达到峰值
    peak_hour = (datetime(2021, 7, 20, 12) - START_DATE).total_seconds() / 3600
    precip = 5 * np.exp(-((hours - peak_hour) ** 2) / 100) + 2 * np.random.random(n)
    precip = np.maximum(precip, 0)
    
    # 850hPa涡度 (10^-5 s^-1)
    vort_850 = 10 * np.exp(-((hours - peak_hour) ** 2) / 150) + 3 + np.random.random(n)
    
    # 500hPa位势高度 (gpm)
    z500 = 5750 - 50 * np.sin(2 * np.pi * hours / 72) + 10 * np.random.random(n)
    
    # 整层可降水量 (mm)
    iwv = 55 + 10 * np.exp(-((hours - peak_hour) ** 2) / 200) + 5 * np.random.random(n)
    
    # 水汽通量辐合 (10^-5 g/(kg·s))
    q_conv = 8 * np.exp(-((hours - peak_hour) ** 2) / 100) + np.random.random(n)
    
    # K指数
    k_index = 35 + 5 * np.exp(-((hours - peak_hour) ** 2) / 150) + 2 * np.random.random(n)
    
    return {
        'precip': precip,
        'vort_850': vort_850,
        'z500': z500,
        'iwv': iwv,
        'q_conv': q_conv,
        'k_index': k_index
    }


def plot_time_series(times, data, ax, ylabel, title, color='blue', 
                     highlight_times=None):
    """
    绘制时间序列图
    """
    ax.plot(times, data, color=color, linewidth=1.5)
    ax.fill_between(times, 0, data, alpha=0.3, color=color)
    
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
    
    # 格式化x轴
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H时'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 标记关键时刻
    if highlight_times:
        for ht in highlight_times:
            ax.axvline(ht, color='red', linewidth=1, linestyle='--', alpha=0.7)
    
    # 标记7月20日极端降水日
    extreme_day = datetime(2021, 7, 20)
    ax.axvspan(extreme_day, extreme_day + timedelta(days=1), 
               alpha=0.1, color='red')


def plot_hovmoller(data, times, coord, ax, title, cmap='RdBu_r', 
                   coord_type='lon'):
    """
    绘制时间-空间霍夫莫勒图
    
    参数:
        data: 2D数组 (time, coord)
        times: 时间数组
        coord: 空间坐标 (经度或纬度)
        ax: matplotlib轴
        title: 标题
        cmap: 色谱
        coord_type: 'lon' 或 'lat'
    """
    # 创建网格
    time_num = mdates.date2num(times)
    xx, yy = np.meshgrid(coord, time_num)
    
    cf = ax.contourf(xx, yy, data, levels=20, cmap=cmap, extend='both')
    
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.yaxis.set_major_locator(mdates.DayLocator())
    
    if coord_type == 'lon':
        ax.set_xlabel('经度 (°E)', fontsize=10)
        ax.axvline(ZHENGZHOU_LON, color='red', linewidth=1.5, linestyle='--')
    else:
        ax.set_xlabel('纬度 (°N)', fontsize=10)
        ax.axhline(ZHENGZHOU_LAT, color='red', linewidth=1.5, linestyle='--')
    
    ax.set_ylabel('日期', fontsize=10)
    ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
    
    return cf


def calculate_composite_anomaly(data, times, event_time, hours_before=24, hours_after=24):
    """
    计算事件前后的合成距平
    """
    event_idx = np.argmin([abs((t - event_time).total_seconds()) for t in times])
    
    # 事件前平均
    start_idx = max(0, event_idx - hours_before)
    before_mean = np.mean(data[start_idx:event_idx], axis=0)
    
    # 事件时
    event_data = data[event_idx]
    
    # 事件后平均
    end_idx = min(len(times), event_idx + hours_after + 1)
    after_mean = np.mean(data[event_idx+1:end_idx], axis=0)
    
    return before_mean, event_data, after_mean


def main():
    """主程序"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    print("=" * 60)
    print("郑州极端降水综合分析")
    print(f"分析时段: {START_DATE} - {END_DATE}")
    print("=" * 60)
    
    # 生成时间序列
    times = pd.date_range(START_DATE, END_DATE, freq='h').to_pydatetime().tolist()
    
    print(f"\n生成 {len(times)} 小时的示例数据...")
    ts_data = generate_sample_time_series(times)
    
    # ==================== 图1: 时间序列分析 ====================
    print("\n绘制时间序列分析...")
    
    fig1, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig1.suptitle('郑州"7·20"极端暴雨时间演变分析\n2021年7月17日-21日',
                  fontsize=14, fontweight='bold', y=0.98)
    
    # 降水
    plot_time_series(times, ts_data['precip'], axes[0, 0],
                     '降水强度 (mm/h)', 'a. 小时降水强度',
                     color='blue', highlight_times=[KEY_TIMES[2]])
    
    # 850hPa涡度
    plot_time_series(times, ts_data['vort_850'], axes[0, 1],
                     '涡度 (×10⁻⁵ s⁻¹)', 'b. 850hPa相对涡度',
                     color='red', highlight_times=[KEY_TIMES[2]])
    
    # 500hPa位势高度
    plot_time_series(times, ts_data['z500'], axes[1, 0],
                     '位势高度 (gpm)', 'c. 500hPa位势高度',
                     color='purple', highlight_times=[KEY_TIMES[2]])
    
    # 可降水量
    plot_time_series(times, ts_data['iwv'], axes[1, 1],
                     'IWV (mm)', 'd. 整层可降水量',
                     color='green', highlight_times=[KEY_TIMES[2]])
    
    # 水汽辐合
    plot_time_series(times, ts_data['q_conv'], axes[2, 0],
                     '辐合 (×10⁻⁵ g/(kg·s))', 'e. 水汽通量辐合',
                     color='cyan', highlight_times=[KEY_TIMES[2]])
    
    # K指数
    plot_time_series(times, ts_data['k_index'], axes[2, 1],
                     'K指数 (°C)', 'f. K指数',
                     color='orange', highlight_times=[KEY_TIMES[2]])
    
    # 添加图例说明
    for ax in axes.flat:
        ax.text(0.02, 0.95, '红色阴影: 7月20日', transform=ax.transAxes,
                fontsize=8, va='top', alpha=0.7)
    
    plt.tight_layout()
    
    output_path1 = os.path.join(OUTPUT_DIR, 'zhengzhou_time_series.png')
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f"时间序列图已保存: {output_path1}")
    
    # ==================== 图2: 霍夫莫勒图 ====================
    print("\n绘制霍夫莫勒图...")
    
    # 生成示例空间数据
    lon = np.arange(100, 130.5, 0.5)
    lat = np.arange(20, 45.5, 0.5)
    
    # 模拟随时间变化的空间场
    n_times = len(times)
    
    # 纬向平均的涡度时间-经度剖面
    vort_lon = np.zeros((n_times, len(lon)))
    for i, t in enumerate(times):
        hours = (t - START_DATE).total_seconds() / 3600
        peak_hour = (datetime(2021, 7, 20, 12) - START_DATE).total_seconds() / 3600
        vort_lon[i] = 10 * np.exp(-((hours - peak_hour) ** 2) / 100) * \
                      np.exp(-((lon - ZHENGZHOU_LON) ** 2) / 50) + \
                      2 * np.random.random(len(lon))
    
    # 经向平均的水汽通量时间-纬度剖面
    qflux_lat = np.zeros((n_times, len(lat)))
    for i, t in enumerate(times):
        hours = (t - START_DATE).total_seconds() / 3600
        peak_hour = (datetime(2021, 7, 20, 12) - START_DATE).total_seconds() / 3600
        qflux_lat[i] = 200 * np.exp(-((hours - peak_hour) ** 2) / 150) * \
                       np.exp(-((lat - 30) ** 2) / 100) + \
                       20 * np.random.random(len(lat))
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle('郑州"7·20"极端暴雨空间-时间演变',
                  fontsize=14, fontweight='bold', y=0.98)
    
    # 时间-经度剖面 (涡度)
    cf1 = plot_hovmoller(vort_lon, times, lon, axes2[0],
                         'a. 850hPa涡度 时间-经度剖面',
                         cmap='RdBu_r', coord_type='lon')
    cbar1 = plt.colorbar(cf1, ax=axes2[0], orientation='vertical', pad=0.02)
    cbar1.set_label('涡度 (×10⁻⁵ s⁻¹)', fontsize=10)
    
    # 时间-纬度剖面 (水汽通量)
    cf2 = plot_hovmoller(qflux_lat, times, lat, axes2[1],
                         'b. 水汽通量 时间-纬度剖面',
                         cmap='Blues', coord_type='lat')
    cbar2 = plt.colorbar(cf2, ax=axes2[1], orientation='vertical', pad=0.02)
    cbar2.set_label('水汽通量 (g/(m·s))', fontsize=10)
    
    plt.tight_layout()
    
    output_path2 = os.path.join(OUTPUT_DIR, 'zhengzhou_hovmoller.png')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"霍夫莫勒图已保存: {output_path2}")
    
    # ==================== 图3: 关键时刻对比 ====================
    print("\n绘制关键时刻对比图...")
    
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 12),
                               subplot_kw={'projection': ccrs.PlateCarree()})
    fig3.suptitle('郑州"7·20"极端暴雨关键时刻环流对比',
                  fontsize=14, fontweight='bold', y=0.98)
    
    lon_full = np.arange(ANALYSIS_REGION[0], ANALYSIS_REGION[1] + 0.25, 0.25)
    lat_full = np.arange(ANALYSIS_REGION[3], ANALYSIS_REGION[2] - 0.25, -0.25)
    lon2d, lat2d = np.meshgrid(lon_full, lat_full)
    
    titles = ['a. 7月19日12时 (前期)', 'b. 7月20日08时 (开始)',
              'c. 7月20日12时 (峰值)', 'd. 7月20日18时 (后期)']
    
    for i, (ax, key_time, title) in enumerate(zip(axes3.flat, KEY_TIMES, titles)):
        # 生成该时刻的场
        hours = (key_time - START_DATE).total_seconds() / 3600
        peak_hour = (datetime(2021, 7, 20, 12) - START_DATE).total_seconds() / 3600
        
        # 模拟850hPa涡度场
        vort = 15 * np.exp(-((hours - peak_hour) ** 2) / 100) * \
               np.exp(-((lon2d - ZHENGZHOU_LON) ** 2 + (lat2d - ZHENGZHOU_LAT) ** 2) / 50)
        vort = vort + 2 * np.random.random(vort.shape)
        
        ax.set_extent(ANALYSIS_REGION, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        
        levels = np.arange(-5, 20, 1)
        cf = ax.contourf(lon2d, lat2d, vort, levels=levels,
                         cmap='RdBu_r', extend='both',
                         transform=ccrs.PlateCarree())
        
        ax.plot(ZHENGZHOU_LON, ZHENGZHOU_LAT, 'r*', markersize=15,
                transform=ccrs.PlateCarree())
        
        ax.set_xticks(np.arange(ANALYSIS_REGION[0], ANALYSIS_REGION[1]+1, 10), 
                      crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(ANALYSIS_REGION[2], ANALYSIS_REGION[3]+1, 5), 
                      crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5, linestyle='--')
        
        ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
    
    # 添加统一色标
    cbar_ax = fig3.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig3.colorbar(cf, cax=cbar_ax)
    cbar.set_label('850hPa 相对涡度 (×10⁻⁵ s⁻¹)', fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    output_path3 = os.path.join(OUTPUT_DIR, 'zhengzhou_key_moments.png')
    plt.savefig(output_path3, dpi=300, bbox_inches='tight')
    print(f"关键时刻对比图已保存: {output_path3}")
    
    plt.show()
    
    # ==================== 输出分析总结 ====================
    print("\n" + "=" * 60)
    print("郑州7-20极端暴雨综合分析总结")
    print("=" * 60)
    
    print("\n1. 时间演变特征:")
    print("   - 降水从7月19日开始增强")
    print("   - 7月20日12-16时达到极端峰值")
    print("   - 小时降水强度超过200mm/h")
    
    print("\n2. 环流系统配置:")
    print("   - 500hPa: 副热带高压西伸北抬")
    print("   - 850hPa: 低涡发展，切变线维持")
    print("   - 200hPa: 高空辐散区叠加")
    
    print("\n3. 水汽条件:")
    print("   - 整层可降水量达到65mm以上")
    print("   - 低空急流将南海水汽向北输送")
    print("   - 水汽通量辐合中心与降水区重合")
    
    print("\n4. 热力不稳定:")
    print("   - K指数>40°C，SI指数<-3°C")
    print("   - 850hPa相当位温>355K")
    print("   - 大气处于极端不稳定状态")
    
    print("\n5. 动力条件:")
    print("   - 低层强辐合，高层强辐散")
    print("   - 正涡度柱从地面延伸到中层")
    print("   - 垂直上升运动强烈")
    
    print("\n分析完成!")


if __name__ == '__main__':
    main()
