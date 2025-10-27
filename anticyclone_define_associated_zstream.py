import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
from tqdm import tqdm

def process_date(date, pressure_groups, stream_groups):
    pres_date = pressure_groups.get(date, pd.DataFrame())
    stream_date = stream_groups.get(date, pd.DataFrame())
    results = []

    if pres_date.empty or stream_date.empty:
        return results

    # 转换为 NumPy 数组
    pres_lons = pres_date['center_lon'].values
    pres_lats = pres_date['center_lat'].values
    pres_values = pres_date['z'].values
    stream_lons = stream_date['center_lon'].values
    stream_lats = stream_date['center_lat'].values
    stream_values = stream_date['stream'].values

    for i, (s_lon, s_lat, s_value) in enumerate(zip(pres_lons, pres_lats, pres_values)):
        # 向量化距离计算
        lon_diff = np.abs(stream_lons - s_lon)
        lat_diff = np.abs(stream_lats - s_lat)
        mask = (lon_diff <= 5) & (lat_diff <= 5)
        nearby_values = stream_values[mask]
        if len(nearby_values) > 0:
            max_idx = np.argmax(np.abs(nearby_values))
            results.append({
                'date': date,
                'pressure_lon': s_lon,
                'pressure_lat': s_lat,
                'pressure_value': s_value,
                'stream_lon': stream_lons[mask][max_idx],
                'stream_lat': stream_lats[mask][max_idx],
                'stream_value': nearby_values[max_idx]
            })
    return results

def main():
    # 文件路径
    pressure_file = 'potential_high_pressure_centers_z_850hpa_tot.csv'
    stream_file = 'potential_high_pressure_centers_stream_850hpa_tot.csv'

    # 读取 CSV 文件，仅加载必要列
    pressure_df = pd.read_csv(pressure_file, usecols=['date', 'center_lon', 'center_lat', 'z'])
    stream_df = pd.read_csv(stream_file, usecols=['date', 'center_lon', 'center_lat', 'stream'])

    # 确保经纬度为浮点数
    pressure_df['center_lon'] = pressure_df['center_lon'].astype(float)
    pressure_df['center_lat'] = pressure_df['center_lat'].astype(float)
    stream_df['center_lon'] = stream_df['center_lon'].astype(float)
    stream_df['center_lat'] = stream_df['center_lat'].astype(float)

    # 按日期分组
    pressure_groups = dict(tuple(pressure_df.groupby('date')))
    stream_groups = dict(tuple(stream_df.groupby('date')))
    dates = stream_df['date'].unique()

    # 使用 ProcessPoolExecutor 进行并行处理
    results = []
    with ThreadPoolExecutor(max_workers=min(os.cpu_count(), len(dates))) as executor:
        process_func = partial(process_date, pressure_groups=pressure_groups, stream_groups=stream_groups)
        for result in tqdm(executor.map(process_func, dates), total=len(dates), desc="处理日期"):
            results.extend(result)

    # 转换为 DataFrame 并按日期排序
    result_df = pd.DataFrame(results).sort_values('date')

    # 保存结果到 CSV
    result_df.to_csv('confirmed_high_pressure_centers_850hPa_zs.csv', index=False)
    print("结果已保存至 'confirmed_high_pressure_centers.csv'")
    print(result_df)

if __name__ == '__main__':
    main()