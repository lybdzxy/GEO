import xarray as xr
import numpy as np
import cupy as cp
import pandas as pd

# 读取CSV文件
csv_path = 'confirmed_high_pressure_centers_slp1000s.csv'
try:
    df = pd.read_csv(csv_path)
except Exception as e:
    df = pd.DataFrame()

# 如果CSV为空，退出
if df.empty:
    df = pd.DataFrame()

# 获取所有唯一日期
dates = df['date'].unique()

# 初始化结果列表
results = []


# 经度处理函数（调整为9列）
def wrap_longitude(pressure_lon, lon_array, delta_phi):
    """处理经度0°/360°跳跃，返回子网格经度、掩码和索引（9列）"""
    half_width = 4 * delta_phi
    lon_min = (pressure_lon - half_width) % 360
    lon_max = (pressure_lon + half_width) % 360
    if lon_min <= lon_max:
        mask = (lon_array >= lon_min) & (lon_array <= lon_max)
    else:
        mask = (lon_array >= lon_min) | (lon_array <= lon_max)
    lon_indices = np.where(mask)[0]
    lon_sub = lon_array[lon_indices]
    angles = ((lon_sub - pressure_lon + 180) % 360 - 180)
    sort_idx = np.argsort(angles)
    lon_sub = lon_sub[sort_idx]
    lon_indices = lon_indices[sort_idx]
    if len(lon_sub) > 9:
        center_idx = len(lon_sub) // 2
        start_idx = max(0, center_idx - 4)
        end_idx = min(len(lon_sub), center_idx + 5)
        lon_sub = lon_sub[start_idx:end_idx]
        lon_indices = lon_indices[start_idx:end_idx]
    return lon_sub, mask, lon_indices


use_gpu = False

# 对每个日期循环处理
for target_date in dates:
    # 筛选当前日期的中心点
    system_centers = df[df['date'] == target_date][['pressure_lon', 'pressure_lat']].values.tolist()

    if not system_centers:
        continue

    # 根据小时选择数据路径
    year = str(target_date)[:4]
    month = str(target_date)[4:6]
    hour = str(target_date)[8:10]
    # 根据小时确定数据路径
    if hour == '00':
        data_path = f'F:/ERA5/hourly/lvl/0z/ERA5_0z_lvl_{year}{month}.nc'
    elif hour == '06':
        data_path = f'F:/ERA5/hourly/lvl/6z/ERA5_6z_lvl_{year}{month}.nc'
    elif hour == '12':
        data_path = f'F:/ERA5/hourly/lvl/12z/ERA5_12z_lvl_{year}{month}.nc'
    elif hour == '18':
        data_path = f'F:/ERA5/hourly/lvl/18z/ERA5_18z_lvl_{year}{month}.nc'
    else:
        continue

    # 加载NetCDF数据
    try:
        ds = xr.open_dataset(data_path)
        lat = ds['latitude'].values
        lon = ds['longitude'].values
    except Exception as e:
        ds = None
        continue

    time_dim = 'time' if 'time' in ds.dims else 'valid_time'
    level_dim = 'level' if 'level' in ds.dims else 'pressure_level'
    time = f"{year}-{str(target_date)[4:6]}-{str(target_date)[6:8]} {hour}:00"
    try:
        z500 = ds['z'].sel({time_dim: time, level_dim: 500})
        z1000 = ds['z'].sel({time_dim: time, level_dim: 1000})
        thickness = z500 - z1000
    except Exception as e:
        ds.close()
        continue

    # 反转纬度（若从北到南）
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        thickness = thickness[::-1, :]

    delta_phi = np.abs(lat[1] - lat[0])

    # 对每个系统中心处理
    for pressure_lon, pressure_lat in system_centers:
        # 纬度范围（仅3行）
        center_lat_idx = np.argmin(np.abs(lat - pressure_lat))
        lat_start_idx = max(0, center_lat_idx - 1)
        lat_end_idx = min(len(lat), center_lat_idx + 2)
        lat_sub = lat[lat_start_idx:lat_end_idx]

        # 动态调整纬度切片
        try:
            thickness_lat_slice = thickness.isel(latitude=slice(lat_start_idx, lat_end_idx))
        except Exception:
            continue

        if len(thickness_lat_slice.latitude) == 0:
            continue

        # 经度环绕（限制为9列）
        lon_sub, lon_mask, lon_indices = wrap_longitude(pressure_lon, lon, delta_phi)

        # 提取3行×9列子网格
        thickness_sub = thickness_lat_slice.values[:, lon_indices]

        if lat_sub[0] > lat_sub[-1]:
            lat_sub = lat_sub[::-1]
            thickness_sub = thickness_sub[::-1, :]

        if thickness_sub.size == 0:
            continue

        # 转换为CuPy
        if use_gpu:
            thickness_sub = cp.array(thickness_sub)
            array_mod = cp
        else:
            array_mod = np

        # 获取网格尺寸
        n_lat, n_lon = thickness_sub.shape

        # 验证子网格是否为3行×9列
        if n_lat != 3 or n_lon != 9:
            continue

        # 找到中心索引
        center_lat_idx = np.argmin(np.abs(lat_sub - pressure_lat))
        center_lon_idx = np.argmin(np.abs(lon_sub - pressure_lon))

        # 中心3x3网格
        center_start_lat = max(0, center_lat_idx - 1)
        center_end_lat = min(n_lat, center_lat_idx + 2)
        center_start_lon = max(0, center_lon_idx - 1)
        center_end_lon = min(n_lon, center_lon_idx + 2)
        center_mean = array_mod.mean(thickness_sub[center_start_lat:center_end_lat, center_start_lon:center_end_lon])

        # 西侧三行9点（列[0:3]）
        west_mean = array_mod.mean(thickness_sub[center_start_lat:center_end_lat, 0:3])

        # 东侧三行9点（列[-3:]）
        east_mean = array_mod.mean(thickness_sub[center_start_lat:center_end_lat, -3:])
        total_mean = (west_mean + east_mean) / 2

        # 判定冷热并存储结果
        system_type = 'c' if center_mean < total_mean else 'w'
        result_row = df[(df['date'] == target_date) &
                        (df['pressure_lon'] == pressure_lon) &
                        (df['pressure_lat'] == pressure_lat)].copy()
        result_row['system_type'] = system_type
        results.append(result_row)

    # 关闭数据集
    ds.close()

# 将结果保存到新的CSV
if results:
    result_df = pd.concat(results, ignore_index=True)
    result_df.to_csv('confirmed_anticyclone_centers_slp1000s_cw.csv', index=False)