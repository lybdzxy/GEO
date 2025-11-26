import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import MultipleLocator
from datetime import datetime
import metpy.calc as mpcalc
from metpy.units import units
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ==================== 0. 字体设置 ====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# =====================================================================

# ==================== 1. 读取轨迹数据 ====================
poi_list_path = 'trajectories_zs.csv'
poi_list = pd.read_csv(poi_list_path)

# ==================== 2. 选取点 ====================
random = False
if random:
    traj_id = np.random.choice(poi_list['trajectory_id'].unique())
    traj_data = poi_list[poi_list['trajectory_id'] == traj_id]
    idx = np.random.randint(len(traj_data))
    row = traj_data.iloc[idx]
    print(f"随机选择：轨迹ID = {traj_id}，第 {idx + 1} 个点")
else:
    traj_id = 7
    date = 1940012606
    row = poi_list[(poi_list['trajectory_id'] == traj_id) & (poi_list['date'] == date)].iloc[0]

center_lon = row['pressure_lon']
center_lat = row['pressure_lat']
date = int(row['date'])
print(f"最终点：轨迹ID: {traj_id} | 时间: {date} | 中心: ({center_lon:.2f}°E, {center_lat:.2f}°N)")

# ==================== 3. 解析时间 ====================
time_str = str(date)
year, month, day, hour = int(time_str[0:4]), int(time_str[4:6]), int(time_str[6:8]), int(time_str[8:10])
target_dt = datetime(year, month, day, hour)

# ==================== 4. 读取 ERA5 数据 ====================
if random:
    data_path = f'F:/ERA5/hourly/lvl/{hour}z/ERA5_{hour}z_lvl_{year}{month:02}.nc'
else:
    data_path = r'E:\GEO\pyproject\example_data\ERA5_6z_lvl_194001.nc'
data = xr.open_dataset(data_path, decode_times=True)
time_dim = 'time' if 'time' in data.dims else 'valid_time'
level_dim = 'level' if 'level' in data.dims else 'pressure_level'

# ==================== 4.1 读取并统一地形数据（关键：统一为 0–360） ====================
dem_path = 'ETOPO_2022_v1_60s_N90W180_bed.nc'
dem = xr.open_dataset(dem_path)
dem = dem.rename({'lon': 'longitude', 'lat': 'latitude'})
dem['longitude'] = (dem.longitude + 360) % 360     # 统一为 0–360
dem = dem.sortby('longitude')

# ==================== 5. 选择时间层 ====================
da = data.sel({time_dim: target_dt})

# ==================== 5.1 只保留 1000–700 hPa ====================
da = da.sel(pressure_level=slice(1000, 700))
# =================================================================

# ==================== 6. 提取变量 + MetPy 转换 w ====================
z = da['z'] / 9.80665
t = da['t']
omega = da['w']
u = da['u']
v = da['v']
elevation = dem['z']
pressure_hpa = da['pressure_level'].values
w_m_s_val = np.full_like(omega.values, np.nan)

for i, p_hpa in enumerate(pressure_hpa):
    p = p_hpa * units.hPa
    t_layer = t.isel(pressure_level=i).values * units.kelvin
    omega_layer = omega.isel(pressure_level=i).values * units('Pa/s')
    w_layer = mpcalc.vertical_velocity(omega_layer, p, t_layer,
                                       mixing_ratio=0 * units('kg/kg'))
    w_m_s_val[i, :, :] = w_layer.magnitude

w_m_s = xr.DataArray(w_m_s_val, coords=omega.coords, dims=omega.dims,
                     name='w_m_s', attrs={'units': 'm/s'})

# ==================== 7. 范围 ±15° (优化处理经度跨越) ====================
extent = 15

def normalize_lon(lon):
    return lon % 360

center_lon = normalize_lon(center_lon)
lon_min = normalize_lon(center_lon - extent)
lon_max = normalize_lon(center_lon + extent)

print(f"中心经度规范化: {center_lon:.2f}°")
print(f"目标经度范围: [{lon_min:.1f}°, {lon_max:.1f}°] (规范化后)")

def select_lon_range(da, lon_min, lon_max):
    lon = da['longitude'].values
    if lon_min < lon_max:
        return da.sel(longitude=slice(lon_min, lon_max))
    else:
        part1 = da.sel(longitude=slice(lon_min, lon[-1]))
        part2 = da.sel(longitude=slice(lon[0], lon_max))
        part2 = part2.assign_coords(longitude=part2['longitude'] + 360)
        return xr.concat([part1, part2], dim='longitude')

lat_slice = slice(center_lat + extent, center_lat - extent)

z = select_lon_range(z, lon_min, lon_max).sel(latitude=lat_slice)
t = select_lon_range(t, lon_min, lon_max).sel(latitude=lat_slice)
u = select_lon_range(u, lon_min, lon_max).sel(latitude=lat_slice)
v = select_lon_range(v, lon_min, lon_max).sel(latitude=lat_slice)
w_m_s = select_lon_range(w_m_s, lon_min, lon_max).sel(latitude=lat_slice)

data_slice = select_lon_range(da, lon_min, lon_max).sel(latitude=lat_slice)

# 地形使用正序纬度
lat_slice_dem = slice(center_lat - extent, center_lat + extent)
elevation = select_lon_range(elevation, lon_min, lon_max).sel(latitude=lat_slice_dem)

print(f"实际提取经度范围: {z.longitude.min().values:.1f}° ~ {z.longitude.max().values:.1f}°")

# ==================== 8. 中心剖面 ====================
center_lon_sel = center_lon if center_lon <= 360 else center_lon - 360

z_lon = z.sel(longitude=center_lon_sel, method='nearest')
t_lon = t.sel(longitude=center_lon_sel, method='nearest')
u_lon = u.sel(longitude=center_lon_sel, method='nearest')
w_lon = w_m_s.sel(longitude=center_lon_sel, method='nearest')

z_lat = z.sel(latitude=center_lat, method='nearest')
t_lat = t.sel(latitude=center_lat, method='nearest')
v_lat = v.sel(latitude=center_lat, method='nearest')
w_lat = w_m_s.sel(latitude=center_lat, method='nearest')

level = 850
z_level = data_slice['z'].sel({level_dim: level})
u_level = data_slice['u'].sel({level_dim: level})
v_level = data_slice['v'].sel({level_dim: level})
lon2d, lat2d = np.meshgrid(data_slice['longitude'].values, data_slice['latitude'].values)
z_vals = z_level.values / 9.80665
u_vals = u_level.values
v_vals = v_level.values
wind_speed = np.sqrt(u_vals ** 2 + v_vals ** 2)

# ==================== 9. colormap ====================
color_list = ['#006400', '#32CD32', '#90EE90', 'white', '#FF69B4', '#FF1493', '#C71585']
bounds = [-0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04]
cmap_omega = ListedColormap(color_list)
norm_omega = BoundaryNorm(bounds, ncolors=len(color_list))
cmap_temp = plt.get_cmap('RdYlBu_r')
cmap_z = plt.get_cmap('terrain')

# ==================== 10. 绘图设置 ====================
fig = plt.figure(figsize=(16, 20))
fig.subplots_adjust(wspace=0.25, hspace=0.3)

ymajorLocator = MultipleLocator(50)
xmajorLocator_lon = MultipleLocator(5)
xmajorLocator_lat = MultipleLocator(10)

# ==================== 最终彻底修复版地形填充函数（直接替换旧的） ====================
def add_terrain_fill(ax, x_fill, is_lon=True):
    """
    稳健地形填充，已彻底解决所有维度不匹配、跨日期线、NaN 问题
    """
    # 构造一个长度完全一致的 DataArray（关键！）
    if is_lon:
        # 经向剖面：经度固定，纬度变化
        coords_dict = {
            'longitude': (['points'], np.full_like(x_fill, center_lon_sel)),
            'latitude':  (['points'], x_fill)
        }
    else:
        # 纬向剖面：纬度固定，经度变化
        coords_dict = {
            'longitude': (['points'], x_fill),
            'latitude':  (['points'], np.full_like(x_fill, center_lat))
        }

    points_da = xr.DataArray(np.zeros(len(x_fill)), dims='points', coords=coords_dict)

    # 线性插值 + nearest 兜底
    elev_1d = elevation.interp(longitude=points_da.longitude, latitude=points_da.latitude,
                               method='linear', kwargs={"fill_value": None})

    elev_values = elev_1d.values

    # 如果有 NaN（跨日期线或边缘），用最近邻补
    if np.isnan(elev_values).any():
        elev_nn = elevation.interp(longitude=points_da.longitude, latitude=points_da.latitude,
                                   method='nearest')
        elev_values = np.nan_to_num(elev_values, nan=elev_nn.values)

    # 保证 x 轴递增
    sort_idx = np.argsort(x_fill)
    x_fill = x_fill[sort_idx]
    elev_values = elev_values[sort_idx]

    # 高度 → 表面气压（0.125 hPa/m 为标准值，4000 m 山会压到 ~500 hPa）
    terrain_h = np.maximum(0, elev_values)
    p_surface = 1013.25 - terrain_h * 0.125

    # 填充灰色地形
    ax.fill_between(x_fill, 1000, p_surface,
                    where=(p_surface < 1000),
                    facecolor='gray', alpha=0.8, zorder=50)

# ==================== 11. 绘图函数（仅替换地形部分，其余完全不动） ====================
def plot_circulation(ax, u_field, w_field, xx, yy, title, is_lon=True):
    # === 原来失败的地形代码全部删除，改用下面两行 ===
    if is_lon:
        add_terrain_fill(ax, z_lon['latitude'].values, is_lon=True)
    else:
        add_terrain_fill(ax, z_lat['longitude'].values, is_lon=False)

    speed = np.sqrt(u_field ** 2 + w_field ** 2)
    u_field = u_field.copy()
    w_field = w_field.copy()
    mask = speed < 0.1
    u_field[mask] = np.nan
    w_field[mask] = np.nan

    ctrf = ax.contourf(xx, yy, w_field,
                       levels=bounds,
                       cmap=cmap_omega,
                       norm=norm_omega,
                       extend='both')
    ax.contour(xx, yy, w_field, levels=np.arange(-0.04, 0.05, 0.01),
               colors='white', linestyles='solid', linewidths=0.4)

    scale_factor = 100
    quiver_scale = 100
    gap_pressure, gap_horizontal = 1, 1

    Q = ax.quiver(xx[::gap_pressure, ::gap_horizontal],
                  yy[::gap_pressure, ::gap_horizontal],
                  u_field[::gap_pressure, ::gap_horizontal],
                  scale_factor * w_field[::gap_pressure, ::gap_horizontal],
                  scale=quiver_scale,
                  pivot='middle',
                  width=0.003,
                  headwidth=6,
                  color='k', zorder=3)

    ax.set_ylim(1000, 700)
    ax.set_yticks(np.arange(700, 1001, 50))
    ax.set_title(title, loc='left', fontsize=14, fontweight='bold')
    ax.tick_params(length=5, width=1.5, labelsize=12)
    ax.yaxis.set_major_locator(ymajorLocator)
    if is_lon:
        ax.set_xlabel('Latitude (°N)')
        ax.xaxis.set_major_locator(xmajorLocator_lon)
    else:
        ax.set_xlabel('Longitude (°E)')
        ax.xaxis.set_major_locator(xmajorLocator_lat)
    return ctrf, Q


def plot_thermo(ax, temp_field, z_field, xx, yy, title, fill_temp=True, is_lon=True):
    # === 原来失败的地形代码全部删除，改用下面两行 ===
    if is_lon:
        add_terrain_fill(ax, z_lon['latitude'].values, is_lon=True)
    else:
        add_terrain_fill(ax, z_lat['longitude'].values, is_lon=False)

    if fill_temp:
        ctrf = ax.contourf(xx, yy, temp_field, levels=np.arange(260, 300, 5),
                           cmap=cmap_temp, extend='both')
        cs = ax.contour(xx, yy, z_field, levels=np.arange(0, 3000, 500),
                        colors='black', linewidths=1.2)
        ax.clabel(cs, inline=True, fontsize=10, fmt='%d')
    else:
        ctrf = ax.contourf(xx, yy, z_field, levels=np.arange(0, 3000, 500),
                           cmap=cmap_z, extend='both')
        cs = ax.contour(xx, yy, temp_field, levels=np.arange(260, 300, 5),
                        colors='red', linewidths=1.0, linestyles='dashed')
        ax.clabel(cs, inline=True, fontsize=10, fmt='%d K')
    ax.set_ylim(1000, 700)
    ax.set_yticks(np.arange(700, 1001, 50))
    ax.set_title(title, loc='left', fontsize=14, fontweight='bold')
    ax.tick_params(length=5, width=1.5, labelsize=12)
    ax.yaxis.set_major_locator(ymajorLocator)
    if is_lon:
        ax.set_xlabel('Latitude (°N)')
        ax.xaxis.set_major_locator(xmajorLocator_lon)
    else:
        ax.set_xlabel('Longitude (°E)')
        ax.xaxis.set_major_locator(xmajorLocator_lat)
    return ctrf


def setup_map(ax, center_lon, center_lat, extent):
    center_lon_proj = center_lon % 360
    proj = ccrs.PlateCarree(central_longitude=center_lon_proj)
    ax.set_extent([center_lon_proj - extent, center_lon_proj + extent,
                   center_lat - extent, center_lat + extent], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.STATES, linestyle=':', alpha=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    return gl


# ==================== 13. 绘图（地图在第一行） ====================
ax1 = fig.add_subplot(3, 2, 1, projection=ccrs.PlateCarree(central_longitude=center_lon % 360))
setup_map(ax1, center_lon, center_lat, extent)

if level == 1000:
    z_levels = np.arange(0, 410, 25)
else:
    z_levels = np.arange(1300, 1710, 25)

cf1 = ax1.contourf(lon2d, lat2d, z_vals, levels=z_levels,
                   cmap='RdYlBu_r', extend='both', transform=ccrs.PlateCarree())
cs1 = ax1.contour(lon2d, lat2d, z_vals, levels=z_levels[::2],
                  colors='black', linewidths=0.7, transform=ccrs.PlateCarree())
ax1.clabel(cs1, inline=True, fontsize=7, fmt='%d')
cbar1 = plt.colorbar(cf1, ax=ax1, orientation='vertical', pad=0.05, shrink=0.7, ticks=z_levels[::2])
cbar1.set_label('Geopotential Height (gpm)')
ax1.plot(center_lon % 360, center_lat, 'o', color='red', markersize=8, markeredgecolor='k', markeredgewidth=0.5,
         transform=ccrs.PlateCarree())
ax1.text((center_lon % 360) + 1, center_lat + 1, 'Center', color='red', fontsize=11, fontweight='bold',
         transform=ccrs.PlateCarree())
ax1.set_title(f'a. {level} hPa 位势高度', loc='left', fontsize=14, fontweight='bold')

ax2 = fig.add_subplot(3, 2, 2, projection=ccrs.PlateCarree(central_longitude=center_lon % 360))
setup_map(ax2, center_lon, center_lat, extent)

wind_levels = np.arange(0, 22, 2)
cf2 = ax2.contourf(lon2d, lat2d, wind_speed, levels=wind_levels,
                   cmap='YlOrRd', extend='max', transform=ccrs.PlateCarree())
cs2 = ax2.contour(lon2d, lat2d, wind_speed, levels=wind_levels[::2],
                  colors='black', linewidths=0.5, alpha=0.7, transform=ccrs.PlateCarree())
step = 1
q = ax2.quiver(lon2d[::step, ::step], lat2d[::step, ::step],
               u_vals[::step, ::step], v_vals[::step, ::step],
               scale=300, color='black', width=0.0025, transform=ccrs.PlateCarree())
cbar2 = plt.colorbar(cf2, ax=ax2, orientation='vertical', pad=0.05, shrink=0.7, ticks=wind_levels[::2])
cbar2.set_label('Wind Speed (m/s)')
ax2.plot(center_lon % 360, center_lat, 'o', color='red', markersize=8, markeredgecolor='k', markeredgewidth=0.5,
         transform=ccrs.PlateCarree())
ax2.text((center_lon % 360) + 1, center_lat + 1, 'Center', color='red', fontsize=11, fontweight='bold',
         transform=ccrs.PlateCarree())
ax2.set_title(f'b. {level} hPa 风场', loc='left', fontsize=14, fontweight='bold')

# ---- 第二行：经向环流 + 纬向环流 ----
xx_lon, yy_lon = np.meshgrid(z_lon['latitude'].values, z_lon['pressure_level'].values)
xx_lat, yy_lat = np.meshgrid(z_lat['longitude'].values, z_lat['pressure_level'].values)

ax3 = fig.add_subplot(3, 2, 3)
ctrf3, Q3 = plot_circulation(ax3, u_lon.values, w_lon.values, xx_lon, yy_lon,
                             f'c. 经向环流 (Lon={center_lon:.1f}°E)', is_lon=True)

ax4 = fig.add_subplot(3, 2, 4)
ctrf4, Q4 = plot_circulation(ax4, v_lat.values, w_lat.values, xx_lat, yy_lat,
                             f'd. 纬向环流 (Lat={center_lat:.1f}°N)', is_lon=False)

# ---- 第三行：温度 + 位势高度剖面 ----
ax5 = fig.add_subplot(3, 2, 5)
ctrf5 = plot_thermo(ax5, t_lon.values, z_lon.values, xx_lon, yy_lon,
                    'e. 温度场 + 位势高度 (经向)', fill_temp=True, is_lon=True)

ax6 = fig.add_subplot(3, 2, 6)
ctrf6 = plot_thermo(ax6, t_lat.values, z_lat.values, xx_lat, yy_lat,
                    'f. 温度场 + 位势高度 (纬向)', fill_temp=True, is_lon=False)

# ==================== 12. 在剖面图（3、4、5、6）X轴下方标注中心经纬度 ====================
center_lat_line = center_lat
center_lon_line = center_lon if center_lon <= 360 else center_lon - 360
lon_in_plot = xx_lat[0, :]
idx_lon = np.argmin(np.abs(lon_in_plot - center_lon_line))
center_lon_plot = lon_in_plot[idx_lon]

ax3.axvline(center_lat_line, color='red', linewidth=1.5, linestyle='--', zorder=5)
ax3.text(center_lat_line, 0.02, f'{center_lat_line:.1f}°N',
         transform=ax3.get_xaxis_transform(), color='red', fontsize=11, fontweight='bold',
         ha='center', va='bottom')

ax4.axvline(center_lon_plot, color='red', linewidth=1.5, linestyle='--', zorder=5)
ax4.text(center_lon_plot, 0.02, f'{center_lon_line:.1f}°E',
         transform=ax4.get_xaxis_transform(), color='red', fontsize=11, fontweight='bold',
         ha='center', va='bottom')

ax5.axvline(center_lat_line, color='red', linewidth=1.5, linestyle='--', zorder=5)
ax5.text(center_lat_line, 0.02, f'{center_lat_line:.1f}°N',
         transform=ax5.get_xaxis_transform(), color='red', fontsize=11, fontweight='bold',
         ha='center', va='bottom')

ax6.axvline(center_lon_plot, color='red', linewidth=1.5, linestyle='--', zorder=5)
ax6.text(center_lon_plot, 0.02, f'{center_lon_line:.1f}°E',
         transform=ax6.get_xaxis_transform(), color='red', fontsize=11, fontweight='bold',
         ha='center', va='bottom')

# ==================== 风矢示例统一放置 ====================
pos_cbar2 = cbar2.ax.get_position()
x_key = pos_cbar2.x1 + 0.02
y_key = pos_cbar2.y0 + pos_cbar2.height / 2
ax2.quiverkey(q, X=x_key, Y=y_key, U=10,
              label='10 m/s', labelpos='E',
              coordinates='figure', fontproperties={'size': 11},
              color='black')

ax3.quiverkey(Q3, X=0.6, Y=1.05, U=10,
              label='水平 10 m/s，垂直放大 100 倍',
              labelpos='E', coordinates='axes', fontproperties={'size': 11})

ax4.quiverkey(Q4, X=0.6, Y=1.05, U=10,
              label='水平 10 m/s，垂直放大 100 倍',
              labelpos='E', coordinates='axes', fontproperties={'size': 11})

# ==================== 14. 色标（精准对齐第二、三行子图中心）===================
pos_row2 = ax3.get_position()
pos_row3 = ax5.get_position()
y_center_row2 = (pos_row2.y0 + pos_row2.y1) / 2
y_center_row3 = (pos_row3.y0 + pos_row3.y1) / 2
height_row = pos_row2.height
cbar_height = height_row * 0.7
cbar_width = 0.015
cbar_left = 0.92

pos1 = [cbar_left, y_center_row2 - cbar_height / 2, cbar_width, cbar_height]
cax1 = fig.add_axes(pos1)
cb1 = fig.colorbar(ctrf3, cax=cax1, orientation='vertical',
                   ticks=bounds, boundaries=bounds, spacing='proportional')
cb1.ax.tick_params(labelsize=11)
cb1.set_label('w (m/s)', fontsize=12)

pos2 = [cbar_left, y_center_row3 - cbar_height / 2, cbar_width, cbar_height]
cax2 = fig.add_axes(pos2)
cb2 = fig.colorbar(ctrf5, cax=cax2, orientation='vertical', ticks=np.arange(260, 300, 5))
cb2.ax.tick_params(labelsize=11)
cb2.set_label('Temperature (K)', fontsize=12)

# ==================== 15. 总标题 & 保存 ====================
fig.suptitle(f'反气旋空间结构分析 (1000–700 hPa)\n'
             f'{year}-{month:02d}-{day:02d} {hour:02d}Z | 轨迹 {traj_id} | '
             f'中心 ({center_lon:.1f}°E, {center_lat:.1f}°N)',
             fontsize=18, y=0.96, fontweight='bold')

output_png = (f'circulation_thermo_700_1000_ID{traj_id}_{year}{month:02d}'
              f'{day:02d}{hour:02d}{"_random" if random else ""}.png')
plt.savefig(output_png, dpi=500, bbox_inches='tight', pad_inches=0.2)
print(f"图像已保存: {output_png}")
plt.show()