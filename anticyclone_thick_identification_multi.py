import xarray as xr
import numpy as np
import cupy as cp
import pandas as pd
import concurrent.futures
import os
from tqdm import tqdm

# 经度处理函数
def wrap_longitude(pressure_lon, lon_array, delta_phi):
    """处理经度0°/360°跳跃，返回子网格经度、掩码和索引（9列）"""
    half_width = 4 * delta_phi
    lon_min = (pressure_lon - half_width) % 360
    lon_max = (pressure_lon + half_width) % 360
    mask = (lon_array >= lon_min) & (lon_array <= lon_max) if lon_min <= lon_max else \
        (lon_array >= lon_min) | (lon_array <= lon_max)
    lon_indices = np.where(mask)[0]
    lon_sub = lon_array[lon_indices]
    angles = ((lon_sub - pressure_lon + 180) % 360 - 180)
    sort_idx = np.argsort(angles)
    lon_sub, lon_indices = lon_sub[sort_idx], lon_indices[sort_idx]

    if len(lon_sub) > 9:
        center_idx = len(lon_sub) // 2
        start_idx = max(0, center_idx - 4)
        end_idx = min(len(lon_sub), center_idx + 5)
        lon_sub, lon_indices = lon_sub[start_idx:end_idx], lon_indices[start_idx:end_idx]
    return lon_sub, mask, lon_indices

# 处理单个中心的函数
def process_center(target_date, pressure_lon, pressure_lat, df_subset, lat, lon, delta_phi, thickness, time_dim, use_gpu=True):
    try:
        time = f"{str(target_date)[:4]}-{str(target_date)[4:6]}-{str(target_date)[6:8]} {str(target_date)[8:10]}:00"

        # 提取 thickness 数据，使用动态的 time_dim
        thickness_t = thickness.sel({time_dim: time})

        # 反转纬度
        if lat[0] > lat[-1]:
            lat = lat[::-1]
            thickness_t = thickness_t[::-1, :]

        # 纬度范围（3行）
        center_lat_idx = np.argmin(np.abs(lat - pressure_lat))
        lat_start_idx = max(0, center_lat_idx - 1)
        lat_end_idx = min(len(lat), center_lat_idx + 2)
        lat_sub = lat[lat_start_idx:lat_end_idx]

        # 经度范围（9列）
        lon_sub, _, lon_indices = wrap_longitude(pressure_lon, lon, delta_phi)
        thickness_sub = thickness_t.isel(latitude=slice(lat_start_idx, lat_end_idx)).values[:, lon_indices]

        # 反转纬度子网格
        if lat_sub[0] > lat_sub[-1]:
            lat_sub = lat_sub[::-1]
            thickness_sub = thickness_sub[::-1, :]

        if thickness_sub.size == 0 or thickness_sub.shape != (3, 9):
            return None

        # 使用 CuPy 加速
        array_mod = cp if use_gpu else np
        thickness_sub = array_mod.array(thickness_sub)

        # 中心3x3网格均值
        center_lat_idx = np.argmin(np.abs(lat_sub - pressure_lat))
        center_lon_idx = np.argmin(np.abs(lon_sub - pressure_lon))
        center_start_lat = max(0, center_lat_idx - 1)
        center_end_lat = min(3, center_lat_idx + 2)
        center_start_lon = max(0, center_lon_idx - 1)
        center_end_lon = min(9, center_lon_idx + 2)
        center_mean = array_mod.mean(thickness_sub[center_start_lat:center_end_lat, center_start_lon:center_end_lon])

        # 西侧和东侧均值
        west_mean = array_mod.mean(thickness_sub[center_start_lat:center_end_lat, 0:3])
        east_mean = array_mod.mean(thickness_sub[center_start_lat:center_end_lat, -3:])
        total_mean = (west_mean + east_mean) / 2
        ratio = center_mean / total_mean
        # 判定冷热
        system_type = 'c' if center_mean < total_mean else 'w'

        # 获取结果
        result_row = df_subset[(df_subset['date'] == target_date) &
                              (df_subset['pressure_lon'] == pressure_lon) &
                              (df_subset['pressure_lat'] == pressure_lat)].copy()
        result_row['system_type'] = system_type
        result_row['ratio'] = ratio
        return result_row
    except Exception as e:
        print(f"Error processing center {target_date}, {pressure_lon}, {pressure_lat}: {e}")
        return None

# 处理单个 NetCDF 文件的函数
def process_file(year, month, hour, dates, df, use_gpu=True, max_workers=12):
    data_path = f'F:/ERA5/hourly/lvl/{int(hour)}z/ERA5_{int(hour)}z_lvl_{year}{month}.nc'
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return []

    try:
        # 加载 NetCDF 文件
        ds = xr.open_dataset(data_path, chunks={'time': 1})  # 使用 Dask 懒加载
        lat, lon = ds['latitude'].values, ds['longitude'].values
        delta_phi = np.abs(lat[1] - lat[0])

        # 计算 thickness（批量），使用动态的 time_dim 和 level_dim
        time_dim = 'time' if 'time' in ds.dims else 'valid_time'
        level_dim = 'level' if 'level' in ds.dims else 'pressure_level'
        z500 = ds['z'].sel({level_dim: 500}).compute()  # 强制计算以释放内存
        z1000 = ds['z'].sel({level_dim: 1000}).compute()
        thickness = z500 - z1000

        # 过滤 df，仅传递与当前 dates 相关的子集
        df_subset = df[df['date'].isin(dates)].copy()

        # 计算该文件中的总中心点数
        total_centers = len(df_subset[['pressure_lon', 'pressure_lat']].drop_duplicates())

        # 使用 ProcessPoolExecutor 并行处理，并添加进度条
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_center, target_date, pressure_lon, pressure_lat, df_subset, lat, lon, delta_phi, thickness, time_dim, use_gpu
                ): (target_date, pressure_lon, pressure_lat)
                for target_date in dates
                for pressure_lon, pressure_lat in df_subset[df_subset['date'] == target_date][['pressure_lon', 'pressure_lat']].values
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)

        ds.close()
        return results
    except Exception as e:
        print(f"Error processing file {data_path}: {e}")
        ds.close()
        return []

# 主函数
def main():
    # 读取 CSV 文件
    csv_path = 'confirmed_high_pressure_centers_850hPa_zs.csv'
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return

    if df.empty:
        print("CSV file is empty.")
        return

    # 按年、月、小时分组
    df['year'] = df['date'].astype(str).str[:4]
    df['month'] = df['date'].astype(str).str[4:6]
    df['hour'] = df['date'].astype(str).str[8:10]
    grouped = df.groupby(['year', 'month', 'hour'])['date'].unique().reset_index()

    # 使用 ProcessPoolExecutor 并行处理文件，并添加进度条
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:  # 限制主进程数
        futures = {
            executor.submit(process_file, row['year'], row['month'], row['hour'], row['date'], df): f"{row['year']}-{row['month']}-{row['hour']}"
            for _, row in grouped.iterrows()
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                           desc="Processing NetCDF files"):
            result = future.result()
            results.extend(result)

    # 合并结果并保存
    if results:
        result_df = pd.concat(results, ignore_index=True)
        result_df.to_csv('confirmed_anticyclone_centers_850hpa_zs_cw.csv', index=False)
        print("Results saved to confirmed_anticyclone_centers_850hpa_zs_cw.csv")
    else:
        print("No results to save.")

if __name__ == '__main__':
    main()