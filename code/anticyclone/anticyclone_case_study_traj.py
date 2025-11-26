import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from datetime import datetime
import metpy.calc as mpcalc
from metpy.units import units
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import warnings

#warnings.filterwarnings('ignore')

# ==================== 参数设置 ====================
OUTPUT_ROOT = r'E:\GEO\pyproject\casestudy\traj'      # 总输出目录
os.makedirs(OUTPUT_ROOT, exist_ok=True)

N_TRAJ = 20                     # ←←←← 这里改你要随机抽取的轨迹数量
SEED   = 42                     # 随机种子，保证可复现

# ==================== 读取轨迹数据 ====================
poi_list_path = 'trajectories_zs.csv'
poi_list = pd.read_csv(poi_list_path)
poi_list['trajectory_id'] = poi_list['trajectory_id'].astype(float)
poi_list['date'] = poi_list['date'].astype(float)

# ==================== 随机挑选完整轨迹 ====================
np.random.seed(SEED)
unique_traj_ids = poi_list['trajectory_id'].unique()
selected_traj_ids = np.random.choice(unique_traj_ids, size=N_TRAJ, replace=False)
print(f"随机挑选了 {len(selected_traj_ids)} 条完整轨迹：{selected_traj_ids}")

records = []
failed = []

# ==================== 保持你原来的所有函数（ew_formatter、ns_formatter、run_full_plot_for_row 等）===================
# ==================== NS / EW 格式化函数 ====================
def ew_formatter(x):
    x = x % 360
    if x > 180:
        return f'{360 - x:.0f}°W'
    elif x == 180:
        return '180°'
    elif x > 0:
        return f'{x:.0f}°E'
    elif x == 0:
        return '0°'
    else:
        return f'{abs(x):.0f}°W'

def ns_formatter(x):
    return f'{abs(x):.0f}°N' if x >= 0 else f'{abs(x):.0f}°S'

def ew_title_formatter(x):
    x = x % 360
    if x > 180:
        return f'{360 - x:.1f}°W'
    elif x == 180:
        return '180°'
    elif x > 0:
        return f'{x:.1f}°E'
    elif x == 0:
        return '0°'
    else:
        return f'{abs(x):.1f}°W'

def ns_title_formatter(x):
    return f'{abs(x):.1f}°N' if x >= 0 else f'{abs(x):.1f}°S'


# ==================== 绘图函数 ====================
def run_full_plot_for_row(traj_id, idx_in_traj, date, center_lon, center_lat, output_png):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # --- 读取轨迹点 ---
    traj_data = poi_list[poi_list['trajectory_id'] == traj_id].copy().reset_index(drop=True)
    row = traj_data.iloc[int(idx_in_traj)]
    center_lon = float(row['pressure_lon'])
    center_lat = float(row['pressure_lat'])
    date_int = int(row['date'])
    print(f"正在处理：轨迹ID = {traj_id}, 时间 = {date_int}, 中心 = ({center_lon:.2f}°E, {center_lat:.2f}°N)")

    # --- 解析时间 ---
    time_str = str(date_int).zfill(10)
    year, month, day, hour = int(time_str[0:4]), int(time_str[4:6]), int(time_str[6:8]), int(time_str[8:10])
    target_dt = datetime(year, month, day, hour)

    # --- 读取 ERA5 数据 ---
    data_path = f'F:/ERA5/hourly/lvl/{hour}z/ERA5_{hour}z_lvl_{year}{month:02d}.nc'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"文件不存在: {data_path}")
    data = xr.open_dataset(data_path, decode_times=True)
    sp_path = f'F:/ERA5/hourly/sfc/fin/ERA5_{hour:02d}z_sfc_{year}{month:02d}_instant.nc'
    sp = xr.open_dataset(sp_path, decode_times=True)

    # --- 统一维度名 ---
    if 'level' in data.dims:
        data = data.rename({'level': 'pressure_level'})
    if 'valid_time' in data.dims:
        data = data.rename({'valid_time': 'time'})

    if 'valid_time' in sp.dims:
        sp = sp.rename({'valid_time': 'time'})

    # --- 选择时间 ---
    da = data.sel(time=target_dt, method='nearest')
    sp = sp.sel(time=target_dt, method='nearest')

    # --- 选择 1000–500 hPa ---
    plev = da['pressure_level']
    if plev.values[0] < plev.values[-1]:
        da = da.sortby('pressure_level', ascending=False)
    mask = (plev >= 500) & (plev <= 1000)
    if not mask.any():
        raise ValueError(f"无 700–1000 hPa 数据")
    da = da.sel(pressure_level=plev[mask])

    # --- 提取变量 ---
    z = da['z'] / 9.80665
    t = da['t']
    omega = da['w']
    u = da['u']
    v = da['v']
    sp = sp['sp']

    # --- 垂直速度 ---
    w_m_s_val = np.full_like(omega.values, np.nan)
    for i, p_hpa in enumerate(da['pressure_level'].values):
        p = p_hpa * units.hPa
        t_layer = t.isel(pressure_level=i).values * units.kelvin
        omega_layer = omega.isel(pressure_level=i).values * units('Pa/s')
        w_layer = mpcalc.vertical_velocity(omega_layer, p, t_layer, mixing_ratio=0 * units('kg/kg'))
        w_m_s_val[i, :, :] = w_layer.magnitude
    w_m_s = xr.DataArray(w_m_s_val, coords=omega.coords, dims=omega.dims, name='w_m_s')

    # --- 经度范围 ±24° ---
    extent = 15
    center_lon_norm = center_lon % 360
    lon_min = (center_lon_norm - extent) % 360
    lon_max = (center_lon_norm + extent) % 360

    def select_lon_range(da, lon_min, lon_max):
        lon = da['longitude'].values
        lon = np.mod(lon, 360)  # 确保 0-360

        if lon_min < lon_max:
            # 不跨日期线
            return da.sel(longitude=slice(lon_min, lon_max))
        else:
            # 跨日期线：先统一坐标到中心在 0-360 范围内
            da_shifted = da.assign_coords(longitude=(da.longitude % 360))
            # 选取两段
            mask1 = da_shifted.longitude >= lon_min
            mask2 = da_shifted.longitude <= lon_max
            selected = da_shifted.where(mask1 | mask2, drop=True)

            # 再把跨日期线那部分 +360 让坐标连续
            lon_new = selected.longitude.copy()
            lon_new = xr.where(lon_new <= lon_max, lon_new + 360, lon_new)
            selected = selected.assign_coords(longitude=lon_new)
            return selected.sortby('longitude')

    lat_slice = slice(center_lat + extent, center_lat - extent)
    z = select_lon_range(z, lon_min, lon_max).sel(latitude=lat_slice)
    t = select_lon_range(t, lon_min, lon_max).sel(latitude=lat_slice)
    u = select_lon_range(u, lon_min, lon_max).sel(latitude=lat_slice)
    v = select_lon_range(v, lon_min, lon_max).sel(latitude=lat_slice)
    w_m_s = select_lon_range(w_m_s, lon_min, lon_max).sel(latitude=lat_slice)
    data_slice = select_lon_range(da, lon_min, lon_max).sel(latitude=lat_slice)
    sp = select_lon_range(sp, lon_min, lon_max).sel(latitude=lat_slice)

    # 关键：把中心经度也转换到同样的 0~360 范围（如果你已经 +360 了，就保持）
    if center_lon_norm < data_slice.longitude.min():  # 说明中心在左边，需要 +360
        center_lon_norm += 360
    z_lon = z.sel(longitude=center_lon_norm, method='nearest')
    t_lon = t.sel(longitude=center_lon_norm, method='nearest')
    u_lon = u.sel(longitude=center_lon_norm, method='nearest')
    w_lon = w_m_s.sel(longitude=center_lon_norm, method='nearest')
    sp_lon = sp.sel(longitude=center_lon_norm, method='nearest')
    z_lat = z.sel(latitude=center_lat, method='nearest')
    t_lat = t.sel(latitude=center_lat, method='nearest')
    v_lat = v.sel(latitude=center_lat, method='nearest')
    w_lat = w_m_s.sel(latitude=center_lat, method='nearest')
    sp_lat = sp.sel(latitude=center_lat, method='nearest')

    # --- 850 hPa 水平场 ---
    level = 850
    if level not in data_slice['pressure_level'].values:
        level = data_slice['pressure_level'].sel(pressure_level=level, method='nearest').values.item()
    z_level = data_slice['z'].sel(pressure_level=level)
    u_level = data_slice['u'].sel(pressure_level=level)
    v_level = data_slice['v'].sel(pressure_level=level)
    lon2d, lat2d = np.meshgrid(data_slice['longitude'].values, data_slice['latitude'].values)
    z_vals = z_level.values / 9.80665
    u_vals = u_level.values
    v_vals = v_level.values
    wind_speed = np.sqrt(u_vals**2 + v_vals**2)

    # --- colormap ---
    color_list = ['#006400', '#32CD32', '#90EE90', 'white', '#FF69B4', '#FF1493', '#C71585']
    bounds = [-0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04]
    cmap_omega = ListedColormap(color_list)
    norm_omega = BoundaryNorm(bounds, ncolors=len(color_list))
    cmap_temp = plt.get_cmap('RdYlBu_r')

    # --- 绘图 ---
    fig = plt.figure(figsize=(6.496, 12))
    fig.subplots_adjust(wspace=0.38, hspace=0.3)

    # --- 子图 1: 位势高度 ---
    ax1 = fig.add_subplot(3, 2, 1, projection=ccrs.PlateCarree(central_longitude=center_lon_norm))
    ax1.set_extent([center_lon_norm - extent, center_lon_norm + extent, center_lat - extent, center_lat + extent], crs=ccrs.PlateCarree())
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax1.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)

    # 手动设置经纬度刻度（整10°/整5°，E/W N/S 格式）
    lon_min_plot = min(data_slice['longitude'].values)
    lon_max_plot = max(data_slice['longitude'].values)
    lat_min_plot = min(data_slice['latitude'].values)
    lat_max_plot = max(data_slice['latitude'].values)

    lon_ticks = np.arange(np.ceil(lon_min_plot / 10) * 10, np.floor(lon_max_plot / 10) * 10 + 1, 10)
    lat_ticks = np.arange(np.ceil(lat_min_plot / 10) * 10, np.floor(lat_max_plot / 10) * 10 + 1, 10)

    ax1.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    ax1.set_xticklabels([ew_formatter(x) for x in lon_ticks])
    ax1.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
    ax1.set_yticklabels([ns_formatter(y) for y in lat_ticks])

    # 网格线（不带标签）
    ax1.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5, linestyle='--')

    # 绘图内容
    z_levels = np.arange(1300, 1710, 25)
    cf1 = ax1.contourf(lon2d, lat2d, z_vals, levels=z_levels, cmap='RdYlBu_r', extend='both', transform=ccrs.PlateCarree())
    cs1 = ax1.contour(lon2d, lat2d, z_vals, levels=z_levels[::2], colors='black', linewidths=0.7, transform=ccrs.PlateCarree())
    ax1.clabel(cs1, inline=True, fontsize=7, fmt='%d')
    cbar1 = plt.colorbar(cf1, ax=ax1, shrink=0.7, pad=0.1, ticks=z_levels[::4], orientation='horizontal')
    cbar1.set_label('Geopotential Height (gpm)')
    ax1.plot(center_lon_norm % 360, center_lat, 'o', color='red', markersize=8, markeredgecolor='k', markeredgewidth=0.5, transform=ccrs.PlateCarree())
    ax1.set_title(f'a.{level}hPa位势高度', loc='left', fontsize=12, fontweight='bold')

    # --- 子图 2: 风场 ---
    ax2 = fig.add_subplot(3, 2, 2, projection=ccrs.PlateCarree(central_longitude=center_lon_norm))
    ax2.set_extent([center_lon_norm - extent, center_lon_norm + extent, center_lat - extent, center_lat + extent], crs=ccrs.PlateCarree())
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax2.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)

    # 手动设置经纬度刻度（同 ax1）
    ax2.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    ax2.set_xticklabels([ew_formatter(x) for x in lon_ticks])
    ax2.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
    ax2.set_yticklabels([ns_formatter(y) for y in lat_ticks])

    # 网格线
    ax2.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5, linestyle='--')

    # 绘图内容
    wind_levels = np.arange(0, 22, 2)
    cf2 = ax2.contourf(lon2d, lat2d, wind_speed, levels=wind_levels, cmap='YlOrRd', extend='max', transform=ccrs.PlateCarree())
    q = ax2.quiver(lon2d[::1, ::1], lat2d[::1, ::1], u_vals[::1, ::1], v_vals[::1, ::1], scale=300, color='black', width=0.0025, transform=ccrs.PlateCarree())
    cbar2 = plt.colorbar(cf2, ax=ax2, shrink=0.7, pad=0.1, ticks=wind_levels[::2], orientation='horizontal')
    cbar2.set_label('Wind Speed (m/s)')
    ax2.plot(center_lon_norm % 360, center_lat, 'o', color='red', markersize=8, markeredgecolor='k', markeredgewidth=0.5, transform=ccrs.PlateCarree())
    ax2.set_title(f'b.{level}hPa风场', loc='left', fontsize=10, fontweight='bold')

    # --- 剖面图网格 ---
    xx_lon, yy_lon = np.meshgrid(z_lon['latitude'].values, z_lon['pressure_level'].values)
    xx_lat, yy_lat = np.meshgrid(z_lat['longitude'].values, z_lat['pressure_level'].values)

    # --- 动态整5/10刻度（仅在数据范围内）---
    lat_min, lat_max = z_lon['latitude'].min().item(), z_lon['latitude'].max().item()
    lon_min_val, lon_max_val = z_lat['longitude'].min().item(), z_lat['longitude'].max().item()

    lat_ticks = np.arange(np.ceil(lat_min / 10) * 10, np.floor(lat_max / 10) * 10 + 1, 10)
    lon_ticks = np.arange(np.ceil(lon_min_val / 10) * 10, np.floor(lon_max_val / 10) * 10 + 1, 10)

    # --- 经向环流 (c) ---
    ax3 = fig.add_subplot(3, 2, 3)
    speed = np.sqrt(u_lon.values**2 + w_lon.values**2)
    mask = speed < 0.1
    u_plot, w_plot = u_lon.values.copy(), w_lon.values.copy()
    u_plot[mask], w_plot[mask] = np.nan, np.nan
    ctrf3 = ax3.contourf(xx_lon, yy_lon, w_plot, levels=bounds, cmap=cmap_omega, norm=norm_omega, extend='both')
    ax3.contour(xx_lon, yy_lon, w_plot, levels=np.arange(-0.04, 0.05, 0.01), colors='white', linewidths=0.4)
    Q3 = ax3.quiver(xx_lon[::1, ::1], yy_lon[::1, ::1], u_plot[::1, ::1], 100 * w_plot[::1, ::1], scale=100, width=0.003, color='k', zorder=3)
    ax3.set_ylim(1000, 500)
    ax3.set_yticks(np.arange(500, 1001, 50))
    ax3.set_title(f'c.经向风场', loc='left', fontsize=10, fontweight='bold')
    ax3.set_xticks(lat_ticks)
    ax3.set_xticklabels([ns_formatter(x) for x in lat_ticks])

    # --- 纬向环流 (d) ---
    ax4 = fig.add_subplot(3, 2, 4)
    speed = np.sqrt((-v_lat.values)**2 + w_lat.values**2)  # 注意：用 -v 的平方
    mask = speed < 0.1
    v_plot, w_plot = v_lat.values.copy(), w_lat.values.copy()
    v_plot[mask], w_plot[mask] = np.nan, np.nan
    ctrf4 = ax4.contourf(xx_lat, yy_lat, w_plot, levels=bounds, cmap=cmap_omega, norm=norm_omega, extend='both')
    ax4.contour(xx_lat, yy_lat, w_plot, levels=np.arange(-0.04, 0.05, 0.01), colors='white', linewidths=0.4)
    Q4 = ax4.quiver(xx_lat[::1, ::1], yy_lat[::1, ::1], -v_plot[::1, ::1], 100 * w_plot[::1, ::1], scale=100, width=0.003, color='k', zorder=3)
    ax4.set_ylim(1000, 500)
    ax4.set_yticks(np.arange(500, 1001, 50))
    ax4.set_title(f'd.纬向风场', loc='left', fontsize=10, fontweight='bold')
    ax4.set_xticks(lon_ticks)
    ax4.set_xticklabels([ew_formatter(x) for x in lon_ticks])


    # --- 温度 + 位势高度 (e) ---
    ax5 = fig.add_subplot(3, 2, 5)
    ctrf5 = ax5.contourf(xx_lon, yy_lon, t_lon.values, levels=np.arange(250, 311, 3), cmap=cmap_temp, extend='both')
    cs5 = ax5.contour(xx_lon, yy_lon, z_lon.values, levels=np.arange(0, 6000, 1000), colors='black', linewidths=1.2)
    ax5.clabel(cs5, inline=True, fontsize=10, fmt='%d')
    ax5.set_ylim(1000, 500)
    ax5.set_yticks(np.arange(500, 1001, 50))
    ax5.set_title('e.经向温压场', loc='left', fontsize=10, fontweight='bold')
    ax5.set_xticks(lat_ticks)
    ax5.set_xticklabels([ns_formatter(x) for x in lat_ticks])

    # ============ 新增：1000-500hPa 厚度（右侧绿色轴）============
    # 提取 500hPa 和 1000hPa 的位势高度（已除以重力加速度 → gpm）
    z500_lon = z_lon.sel(pressure_level=500).values  # 1D array along latitude
    z1000_lon = z_lon.sel(pressure_level=1000).values
    thickness_lon = z500_lon - z1000_lon  # 厚度（gpm）

    ax5_thick = ax5.twinx()  # 右侧新轴
    line_thk_e = ax5_thick.plot(z_lon['latitude'].values, thickness_lon,
                                color='#006400', linewidth=3, label='1000-500hPa 厚度')[0]

    ax5_thick.tick_params(axis='y', colors='#006400')
    ax5_thick.set_ylim(5200, 5900)  # 根据全球/季节调整，夏季可到5800+
    ax5_thick.set_yticks(np.arange(5200, 5901, 100))
    # --- 温度 + 位势高度 (f) ---
    ax6 = fig.add_subplot(3, 2, 6)
    ctrf6 = ax6.contourf(xx_lat, yy_lat, t_lat.values, levels=np.arange(250, 311, 3), cmap=cmap_temp, extend='both')
    cs6 = ax6.contour(xx_lat, yy_lat, z_lat.values, levels=np.arange(0, 6000, 1000), colors='black', linewidths=1.2)
    ax6.clabel(cs6, inline=True, fontsize=10, fmt='%d')
    ax6.set_ylim(1000, 500)
    ax6.set_yticks(np.arange(500, 1001, 50))
    ax6.set_title('f.纬向温压场', loc='left', fontsize=10, fontweight='bold')
    ax6.set_xticks(lon_ticks)
    ax6.set_xticklabels([ew_formatter(x) for x in lon_ticks])

    # ============ 新增：1000-500hPa 厚度（右侧绿色轴）============
    z500_lat = z_lat.sel(pressure_level=500).values
    z1000_lat = z_lat.sel(pressure_level=1000).values
    thickness_lat = z500_lat - z1000_lat

    ax6_thick = ax6.twinx()
    line_thk_f = ax6_thick.plot(z_lat['longitude'].values, thickness_lat,
                                color='#006400', linewidth=3, label='1000-500hPa 厚度')[0]

    ax6_thick.tick_params(axis='y', colors='#006400')
    ax6_thick.set_ylim(5200, 5900)
    ax6_thick.set_yticks(np.arange(5200, 5901, 100))
    # --- 中心线标注前调整中心经度（修复跨越0°问题） ---
    lon_min_val = z_lat['longitude'].min().item()
    lon_max_val = z_lat['longitude'].max().item()
    adjusted_center_lon = center_lon_norm
    if lon_min > lon_max and center_lon_norm <= lon_max:
        adjusted_center_lon += 360

    # --- 中心线标注 ---
    for ax, xval, label in [
        (ax3, center_lat, ns_formatter(center_lat)),
        (ax5, center_lat, ns_formatter(center_lat)),
        (ax4, adjusted_center_lon, ew_formatter(center_lon_norm)),
        (ax6, adjusted_center_lon, ew_formatter(center_lon_norm))
    ]:
        ax.axvline(xval, color='red', linewidth=1.5, linestyle='--', zorder=5)

    # --- 风矢说明 ---
    pos_cbar2 = cbar2.ax.get_position()
    x_key = pos_cbar2.x1 + 0.05
    y_key = pos_cbar2.y0 + pos_cbar2.height / 2
    ax2.quiverkey(q, X=x_key, Y=y_key, U=10, label='10 m/s', labelpos='E', coordinates='figure', fontproperties={'size': 10}, color='black')
    ax3.quiverkey(Q3, X=0.05, Y=-0.15, U=10, label='水平 10 m/s，垂直放大 100 倍', labelpos='E', coordinates='axes', fontproperties={'size': 10})
    ax4.quiverkey(Q4, X=0.05, Y=-0.15, U=10, label='水平 10 m/s，垂直放大 100 倍', labelpos='E', coordinates='axes', fontproperties={'size': 10})

    # --- 色标 ---
    pos_row2 = ax3.get_position()
    pos_row3 = ax5.get_position()
    y_center_row2 = (pos_row2.y0 + pos_row2.y1) / 2
    y_center_row3 = (pos_row3.y0 + pos_row3.y1) / 2
    height_row = pos_row2.height
    cbar_height = height_row * 0.7
    cbar_width = 0.015
    cbar_left = 0.97

    cax1 = fig.add_axes([cbar_left, y_center_row2 - cbar_height / 2, cbar_width, cbar_height])
    cb1 = fig.colorbar(ctrf3, cax=cax1, orientation='vertical', ticks=bounds)
    cb1.ax.tick_params(labelsize=8)
    #cb1.set_label('w (m/s)', fontsize=10)

    cax2 = fig.add_axes([cbar_left, y_center_row3 - cbar_height / 2, cbar_width, cbar_height])
    cb2 = fig.colorbar(ctrf5, cax=cax2, ticks=np.arange(250, 311, 6))
    cb2.ax.tick_params(labelsize=8)
    #cb2.set_label('Temperature (K)', fontsize=10)
    # ============== 完美地形填充（一劳永逸版）==============
    sp_hpa_lon = sp_lon.values / 100.0
    sp_hpa_lat = sp_lat.values / 100.0
    def add_terrain_bar(ax, coords, sp_hpa, coor):
        if coor =='lon':
            S = 1
        elif coor =='lat':
            S = -1
        coords = np.asarray(coords)
        sp_hpa = np.asarray(sp_hpa)

        # 计算每根柱子的宽度（保持你原来的聪明做法）
        width = np.gradient(coords)  # 可能不均匀
        width = np.abs(width)

        bars = []
        for i, (x, h, w) in enumerate(zip(coords, 1050 - sp_hpa, width)):
            if i == 0:
                # 最左边柱子：只画右半部分（向右对齐）
                bar = ax.bar(x, h, width=-w / 2 * S , bottom=sp_hpa[i],
                             align='edge',  # 关键！从 x 位置向右画
                             color='gray', alpha=1, zorder=10)
            elif i == len(coords) - 1:
                # 最右边柱子：只画左半部分（向左对齐 → 用负宽度）
                bar = ax.bar(x, h, width=w / 2 * S, bottom=sp_hpa[i],
                             align='edge',  # 负宽度 + edge = 从 x 向左画
                             color='gray', alpha=1, zorder=10)
            else:
                # 中间柱子正常画
                bar = ax.bar(x, h, width=w, bottom=sp_hpa[i],
                             color='gray', alpha=1, zorder=10)
            bars.extend(bar)

        return bars

    # 使用
    add_terrain_bar(ax3, z_lon['latitude'].values, sp_hpa_lon, 'lon')
    add_terrain_bar(ax5, z_lon['latitude'].values, sp_hpa_lon, 'lon')
    add_terrain_bar(ax4, z_lat['longitude'].values, sp_hpa_lat, 'lat')
    add_terrain_bar(ax6, z_lat['longitude'].values, sp_hpa_lat, 'lat')
    # ====================================================
    # --- 总标题 ---
    latstr = ns_title_formatter(center_lat)
    lonstr = ew_title_formatter(center_lon)
    fig.suptitle(f'反气旋空间结构分析 (1000–500 hPa)\n{year}-{month:02d}-{day:02d} {hour:02d}Z | ({lonstr},{latstr})', fontsize=16, y=0.96)

    plt.savefig(output_png, dpi=500, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"图像已保存: {output_png}")



# ==================== 主循环：逐条轨迹处理 ====================
for traj_id in selected_traj_ids:
    traj_data = poi_list[poi_list['trajectory_id'] == traj_id].copy().reset_index(drop=True)
    if len(traj_data) == 0:
        continue

    # 为当前轨迹创建独立文件夹
    traj_dir = os.path.join(OUTPUT_ROOT, f'traj_{int(traj_id)}')
    os.makedirs(traj_dir, exist_ok=True)

    # 按时间（date）排序，确保图片序号是时间顺序
    traj_data = traj_data.sort_values('date').reset_index(drop=True)

    print(f"\n正在处理轨迹 ID={traj_id}，共 {len(traj_data)} 个时间点，保存至 {traj_dir}")

    for idx_in_traj, row in traj_data.iterrows():
        date       = float(row['date'])
        center_lon = float(row['pressure_lon'])
        center_lat = float(row['pressure_lat'])

        time_str = str(int(date)).zfill(10)
        year  = int(time_str[0:4])
        month = int(time_str[4:6])
        day   = int(time_str[6:8])
        hour  = int(time_str[8:10])

        # 序号从 001 开始，宽度 3 位（如果轨迹很长可改成 4）
        seq_str = f"{idx_in_traj+1:03d}"

        output_png = os.path.join(traj_dir,
            f"{seq_str}_{year}{month:02d}{day:02d}{hour:02d}_ID{traj_id}.png")

        try:
            run_full_plot_for_row(
                traj_id=traj_id,
                idx_in_traj=idx_in_traj,      # 虽然函数内部已经重新取行，但传过去不影响
                date=date,
                center_lon=center_lon,
                center_lat=center_lat,
                output_png=output_png
            )
            records.append({
                'traj_id': traj_id,
                'seq': idx_in_traj + 1,
                'date': date,
                'year': year, 'month': month, 'day': day, 'hour': hour,
                'center_lon': center_lon,
                'center_lat': center_lat,
                'png_file': os.path.basename(output_png),
                'png_path': output_png
            })
            print(f"  {seq_str} 保存成功")
        except Exception as e:
            print(f"  [ERROR] {seq_str} 失败: {e}")
            failed.append({
                'traj_id': traj_id,
                'seq': idx_in_traj + 1,
                'date': date,
                'error': str(e)
            })

# ==================== 保存记录 ====================
record_df = pd.DataFrame(records)
record_df.to_csv(os.path.join(OUTPUT_ROOT, 'all_selected_points.csv'), index=False, encoding='utf-8-sig')

if failed:
    failed_df = pd.DataFrame(failed)
    failed_df.to_csv(os.path.join(OUTPUT_ROOT, 'failed_points.csv'), index=False, encoding='utf-8-sig')
    print(f"\n完成！共处理 {len(selected_traj_ids)} 条轨迹，失败 {len(failed)} 个点")
else:
    print(f"\n完成！共处理 {len(selected_traj_ids)} 条轨迹，全部成功！")

print(f"结果保存在：{OUTPUT_ROOT}")
print("   每个轨迹一个子文件夹，图片按 001_、002_ … 顺序排列")