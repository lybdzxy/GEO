"""
郑州极端降水垂直廓线分析
Vertical profile analysis for Zhengzhou extreme precipitation event (2021.7.20)

功能：
1. 温度、湿度、风速垂直廓线
2. 温度露点差分析
3. 垂直风切变分析

使用ERA5数据进行分析
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import os
from datetime import datetime

# ==================== 配置参数 ====================
DATA_DIR = r'E:/ERA5/hourly'  # ERA5数据目录
OUTPUT_DIR = r'E:/GEO/pyproject/casestudy/zhengzhou'  # 输出目录

# 郑州坐标
ZHENGZHOU_LON = 113.65
ZHENGZHOU_LAT = 34.76

# 剖面范围 (经向和纬向各±10度)
PROFILE_EXTENT = 10

# 分析时间
ANALYSIS_DATE = datetime(2021, 7, 20, 12)


def load_era5_profile_data(data_dir, date):
    """
    加载ERA5廓线数据
    
    参数:
        data_dir: 数据目录
        date: 分析日期
    
    返回:
        数据字典
    """
    date_str = date.strftime('%Y-%m-%d')
    
    data = {}
    
    try:
        # 尝试加载气压层数据
        u_path = os.path.join(data_dir, f'ERA5_{date_str}_pl_u.nc')
        v_path = os.path.join(data_dir, f'ERA5_{date_str}_pl_v.nc')
        q_path = os.path.join(data_dir, f'ERA5_{date_str}_pl_q.nc')
        
        # 检查文件是否存在
        paths_exist = all(os.path.exists(p) for p in [u_path, v_path, q_path])
        
        if paths_exist:
            u_data = xr.open_dataset(u_path)
            v_data = xr.open_dataset(v_path)
            q_data = xr.open_dataset(q_path)
            
            data['u'] = u_data
            data['v'] = v_data
            data['q'] = q_data
            
            return data
    except Exception as e:
        print(f"数据加载错误: {e}")
    
    return None


def calculate_dewpoint_from_q(q, p):
    """
    从比湿计算露点温度
    
    参数:
        q: 比湿 (kg/kg)
        p: 气压 (hPa)
    
    返回:
        露点温度 (K)
    """
    # 计算水汽压
    e = (q * p * 100) / (0.622 + 0.378 * q)  # Pa
    e_mb = e / 100  # hPa
    
    # 防止对数计算出错
    e_mb = np.maximum(e_mb, 0.01)
    
    # 计算露点温度 (Magnus公式)
    a = 17.27
    b = 237.7
    gamma = np.log(e_mb / 6.112)
    td_c = (b * gamma) / (a - gamma)
    td_k = td_c + 273.15
    
    return td_k


def calculate_vertical_velocity_from_omega(omega, p, t):
    """
    将垂直速度从Pa/s转换为m/s
    
    参数:
        omega: 垂直气压速度 (Pa/s)
        p: 气压 (Pa)
        t: 温度 (K)
    
    返回:
        垂直速度 (m/s)
    """
    # 理想气体常数
    R = 287.05  # J/(kg·K)
    g = 9.80665  # m/s²
    
    # 密度
    rho = p / (R * t)
    
    # 转换 w = -omega / (rho * g)
    w = -omega / (rho * g)
    
    return w


def plot_skew_t_lite(t, td, p, ax):
    """
    简化版的温度-露点廓线图
    
    参数:
        t: 温度 (K)
        td: 露点温度 (K)
        p: 气压 (hPa)
        ax: matplotlib轴对象
    """
    # 转换为摄氏度
    t_c = t - 273.15
    td_c = td - 273.15
    
    # 绘制温度廓线
    ax.plot(t_c, p, 'r-', linewidth=2, label='温度')
    ax.plot(td_c, p, 'b--', linewidth=2, label='露点')
    
    # 设置气压轴
    ax.set_yscale('log')
    ax.set_ylim(1000, 100)
    ax.set_yticks([1000, 925, 850, 700, 500, 300, 200, 100])
    ax.set_yticklabels(['1000', '925', '850', '700', '500', '300', '200', '100'])
    
    # 温度范围
    ax.set_xlim(-80, 40)
    ax.set_xlabel('温度 (°C)', fontsize=10)
    ax.set_ylabel('气压 (hPa)', fontsize=10)
    
    # 添加等温线参考
    for temp in range(-80, 50, 10):
        ax.axvline(temp, color='gray', linewidth=0.3, alpha=0.5)
    
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_wind_profile(u, v, p, ax):
    """
    绘制风速风向廓线
    
    参数:
        u, v: 风速分量 (m/s)
        p: 气压 (hPa)
        ax: matplotlib轴对象
    """
    # 计算风速
    ws = np.sqrt(u**2 + v**2)
    
    # 计算风向 (气象风向)
    wd = (270 - np.rad2deg(np.arctan2(v, u))) % 360
    
    # 创建双轴
    ax2 = ax.twiny()
    
    # 绘制风速
    ax.plot(ws, p, 'b-', linewidth=2, label='风速')
    ax.set_xlim(0, 50)
    ax.set_xlabel('风速 (m/s)', fontsize=10, color='blue')
    ax.tick_params(axis='x', colors='blue')
    
    # 绘制风向
    ax2.plot(wd, p, 'g-', linewidth=2, label='风向')
    ax2.set_xlim(0, 360)
    ax2.set_xlabel('风向 (°)', fontsize=10, color='green')
    ax2.tick_params(axis='x', colors='green')
    
    # 设置气压轴
    ax.set_yscale('log')
    ax.set_ylim(1000, 100)
    ax.set_yticks([1000, 925, 850, 700, 500, 300, 200, 100])
    ax.set_yticklabels(['1000', '925', '850', '700', '500', '300', '200', '100'])
    ax.set_ylabel('气压 (hPa)', fontsize=10)
    
    ax.grid(True, alpha=0.3)
    
    # 添加图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)


def plot_humidity_profile(q, p, ax):
    """
    绘制比湿廓线
    
    参数:
        q: 比湿 (kg/kg)
        p: 气压 (hPa)
        ax: matplotlib轴对象
    """
    # 转换为g/kg
    q_gkg = q * 1000
    
    ax.plot(q_gkg, p, 'b-', linewidth=2)
    
    ax.set_xlabel('比湿 (g/kg)', fontsize=10)
    ax.set_ylabel('气压 (hPa)', fontsize=10)
    
    ax.set_yscale('log')
    ax.set_ylim(1000, 100)
    ax.set_yticks([1000, 925, 850, 700, 500, 300, 200, 100])
    ax.set_yticklabels(['1000', '925', '850', '700', '500', '300', '200', '100'])
    
    ax.set_xlim(0, 20)
    ax.grid(True, alpha=0.3)


def main():
    """主程序"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    print("=" * 60)
    print("郑州极端降水垂直廓线分析")
    print(f"分析时间: {ANALYSIS_DATE}")
    print(f"分析位置: ({ZHENGZHOU_LON}°E, {ZHENGZHOU_LAT}°N)")
    print("=" * 60)
    
    # 加载数据
    print("\n加载ERA5数据...")
    data = load_era5_profile_data(DATA_DIR, ANALYSIS_DATE)
    
    if data is None:
        print("数据加载失败，使用示例数据进行演示...")
        # 创建示例数据
        p_levels = np.array([1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
                            750, 700, 650, 600, 550, 500, 450, 400, 350, 300,
                            250, 200, 150, 100])
        
        # 示例温度廓线 (标准大气)
        t0 = 303  # 地面温度 (K)
        lapse_rate = 6.5 / 1000  # K/m
        t = t0 - lapse_rate * (1013.25 - p_levels) * 8  # 简化的温度廓线
        t = np.maximum(t, 200)  # 最低温度
        
        # 示例比湿廓线
        q = 0.018 * np.exp(-((1000 - p_levels) / 300))  # 随高度指数递减
        q = np.maximum(q, 0.0001)
        
        # 计算露点
        td = calculate_dewpoint_from_q(q, p_levels)
        
        # 示例风廓线
        u = 5 + 0.1 * (1000 - p_levels)  # 风速随高度增加
        v = 2 + 0.05 * (1000 - p_levels)
        
        # 示例垂直剖面数据
        lon = np.arange(ZHENGZHOU_LON - PROFILE_EXTENT, ZHENGZHOU_LON + PROFILE_EXTENT + 0.25, 0.25)
        lat = np.arange(ZHENGZHOU_LAT - PROFILE_EXTENT, ZHENGZHOU_LAT + PROFILE_EXTENT + 0.25, 0.25)
        
        # 创建3D数据 (简化)
        lon2d, p2d = np.meshgrid(lon, p_levels)
        lat2d, _ = np.meshgrid(lat, p_levels)
        
        u_3d = 10 + 0.05 * (1000 - p2d) * np.cos(np.deg2rad(lon2d - ZHENGZHOU_LON) * 5)
        v_3d = 5 + 0.03 * (1000 - p2d) * np.sin(np.deg2rad(lon2d - ZHENGZHOU_LON) * 3)
        
    else:
        # 从实际数据提取郑州位置的廓线
        u_data = data['u']
        v_data = data['v']
        q_data = data['q']
        
        # 统一维度名称
        for ds in [u_data, v_data, q_data]:
            if 'level' in ds.dims:
                ds = ds.rename({'level': 'pressure_level'})
        
        # 选择时间
        if 'time' in u_data.dims:
            u_data = u_data.sel(time=ANALYSIS_DATE, method='nearest')
            v_data = v_data.sel(time=ANALYSIS_DATE, method='nearest')
            q_data = q_data.sel(time=ANALYSIS_DATE, method='nearest')
        
        # 提取郑州位置的廓线数据
        u = u_data['u'].sel(longitude=ZHENGZHOU_LON, latitude=ZHENGZHOU_LAT, method='nearest').values
        v = v_data['v'].sel(longitude=ZHENGZHOU_LON, latitude=ZHENGZHOU_LAT, method='nearest').values
        q = q_data['q'].sel(longitude=ZHENGZHOU_LON, latitude=ZHENGZHOU_LAT, method='nearest').values
        
        p_levels = u_data.pressure_level.values
        
        # 这里需要温度数据来计算露点，如果没有则使用简化方法
        t = 303 - 0.065 * (1013.25 - p_levels) * 8  # 假设的温度廓线
        td = calculate_dewpoint_from_q(q, p_levels)
    
    # ==================== 绘图 ====================
    print("\n绘制垂直廓线图...")
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'郑州"7·20"极端暴雨垂直结构分析\n{ANALYSIS_DATE.strftime("%Y年%m月%d日 %H时")} (UTC)\n'
                 f'位置: ({ZHENGZHOU_LON}°E, {ZHENGZHOU_LAT}°N)',
                 fontsize=14, fontweight='bold', y=0.98)
    
    # 子图1: 温度-露点廓线 (简化版)
    ax1 = fig.add_subplot(2, 3, 1)
    plot_skew_t_lite(t, td, p_levels, ax1)
    ax1.set_title('a. 温度-露点廓线', loc='left', fontsize=12, fontweight='bold')
    
    # 子图2: 风速风向廓线
    ax2 = fig.add_subplot(2, 3, 2)
    plot_wind_profile(u, v, p_levels, ax2)
    ax2.set_title('b. 风速风向廓线', loc='left', fontsize=12, fontweight='bold')
    
    # 子图3: 比湿廓线
    ax3 = fig.add_subplot(2, 3, 3)
    plot_humidity_profile(q, p_levels, ax3)
    ax3.set_title('c. 比湿廓线', loc='left', fontsize=12, fontweight='bold')
    
    # 子图4: 温度露点差廓线
    ax4 = fig.add_subplot(2, 3, 4)
    t_td_diff = t - td
    ax4.plot(t_td_diff, p_levels, 'g-', linewidth=2)
    ax4.axvline(0, color='red', linewidth=1, linestyle='--')
    ax4.set_xlabel('温度露点差 (K)', fontsize=10)
    ax4.set_ylabel('气压 (hPa)', fontsize=10)
    ax4.set_yscale('log')
    ax4.set_ylim(1000, 100)
    ax4.set_yticks([1000, 925, 850, 700, 500, 300, 200, 100])
    ax4.set_yticklabels(['1000', '925', '850', '700', '500', '300', '200', '100'])
    ax4.set_xlim(-5, 40)
    ax4.grid(True, alpha=0.3)
    # 填充饱和区域
    ax4.fill_betweenx(p_levels, 0, t_td_diff, where=(t_td_diff < 5),
                      alpha=0.3, color='blue', label='近饱和层')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.set_title('d. 温度露点差廓线', loc='left', fontsize=12, fontweight='bold')
    
    # 子图5: 垂直风切变
    ax5 = fig.add_subplot(2, 3, 5)
    # 计算风切变
    du_dp = np.gradient(u, p_levels)
    dv_dp = np.gradient(v, p_levels)
    shear = np.sqrt(du_dp**2 + dv_dp**2) * 1000  # 转换单位
    
    ax5.plot(shear, p_levels, 'purple', linewidth=2)
    ax5.set_xlabel('垂直风切变 (m/s per 100hPa)', fontsize=10)
    ax5.set_ylabel('气压 (hPa)', fontsize=10)
    ax5.set_yscale('log')
    ax5.set_ylim(1000, 100)
    ax5.set_yticks([1000, 925, 850, 700, 500, 300, 200, 100])
    ax5.set_yticklabels(['1000', '925', '850', '700', '500', '300', '200', '100'])
    ax5.grid(True, alpha=0.3)
    ax5.set_title('e. 垂直风切变', loc='left', fontsize=12, fontweight='bold')
    
    # 子图6: 风矢量随高度变化
    ax6 = fig.add_subplot(2, 3, 6)
    # 选取关键层次的风矢量
    key_levels = [1000, 925, 850, 700, 500, 300, 200]
    for i, lev in enumerate(key_levels):
        if lev in p_levels:
            idx = np.where(p_levels == lev)[0][0]
            ax6.quiver(0, i, u[idx], v[idx], scale=50, width=0.02, 
                      color=plt.cm.viridis(i/len(key_levels)))
            ax6.text(-0.5, i, f'{lev}hPa', fontsize=9, va='center')
    
    ax6.set_xlim(-1, 3)
    ax6.set_ylim(-0.5, len(key_levels) - 0.5)
    ax6.set_xlabel('风矢量', fontsize=10)
    ax6.set_title('f. 各层风矢量', loc='left', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    # 添加风矢量比例尺
    ax6.quiver(2, 0, 10, 0, scale=50, width=0.02, color='black')
    ax6.text(2, -0.3, '10 m/s', fontsize=9, ha='center')
    
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(OUTPUT_DIR, 'zhengzhou_vertical_profile.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n图像已保存: {output_path}")
    
    plt.show()
    
    # ==================== 输出关键参数 ====================
    print("\n" + "=" * 60)
    print("关键廓线参数:")
    print("=" * 60)
    
    # 低层湿度
    if 850 in p_levels:
        idx_850 = np.where(p_levels == 850)[0][0]
        print(f"850hPa 比湿: {q[idx_850]*1000:.2f} g/kg")
        print(f"850hPa 温度露点差: {t[idx_850]-td[idx_850]:.2f} K")
    
    # 低空急流
    if 850 in p_levels:
        ws_850 = np.sqrt(u[idx_850]**2 + v[idx_850]**2)
        print(f"850hPa 风速: {ws_850:.1f} m/s")
    
    # 0-6km风切变 (近似)
    if 500 in p_levels and 1000 in p_levels:
        idx_500 = np.where(p_levels == 500)[0][0]
        idx_1000 = np.where(p_levels == 1000)[0][0]
        bulk_shear = np.sqrt((u[idx_500]-u[idx_1000])**2 + (v[idx_500]-v[idx_1000])**2)
        print(f"1000-500hPa 总体风切变: {bulk_shear:.1f} m/s")
    
    print("\n分析完成!")


if __name__ == '__main__':
    main()
