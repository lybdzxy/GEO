import xarray as xr
from xinvert import invert_Poisson
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import warnings
from contextlib import redirect_stdout

# 忽略非关键警告
warnings.filterwarnings("ignore")

hour = [0, 6, 12, 18]

def chunk_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def process_time_step_chunk(times, vo_data, iParams, time_dim, area):
    """处理一组时间步的流函数计算，重定向invert_Poisson的输出到日志"""
    results = []
    for t in times:
        if t is not None:
            with open(os.devnull, 'w') as fnull:
                with redirect_stdout(fnull):  # 抑制invert_Poisson的控制台输出
                    vo = vo_data.sel({time_dim: t})
                    # 修正涡度，确保全球积分 = 0，直接修改 vo
                    vo -= (vo * area).sum() / area.sum()
                    # 计算流函数
                    sf = invert_Poisson(vo, dims=['latitude', 'longitude'], iParams=iParams)
                    # 去流函数均值，使正负值对称
                    sf = sf.expand_dims({time_dim: [t]})
                    results.append(sf)
                    # 清理临时变量
                    del vo, sf
    return results

def process_month(year, month, hour):
    """处理指定年月的全部时间步"""
    out_path = f"F:/ERA5/hourly/lvl/stream/ERA5_stream_850hpa_{year}{month:02d}.nc"
    if os.path.exists(out_path):
        print(f"已存在: {out_path}，跳过")
        return

    all_sf = []
    for h in hour:
        data_path = f'F:/ERA5/hourly/lvl/{h}z/ERA5_{h}z_lvl_{year}{month:02d}.nc'
        if not os.path.exists(data_path):
            print(f"缺少文件: {data_path}")
            continue

        # 使用with语句确保数据集关闭
        with xr.open_dataset(data_path) as ds:
            ds = ds.sortby('latitude')
            lat = ds['latitude']
            lon = ds['longitude']
            time_dim = 'time' if 'time' in ds.dims else 'valid_time'
            level_dim = 'level' if 'level' in ds.dims else 'pressure_level'

            # 计算面积权重（仅一次）
            area = np.cos(np.deg2rad(lat))  # 1D: [latitude]
            area = area.expand_dims(longitude=lon)  # 扩展为 2D: [latitude, longitude]

            # 反演参数（保持原设置）
            iParams = {
                'BCs': ['extend', 'periodic'],
                'mxLoop': 1000,
                'tolerance': 1e-6,
            }

            # 选择1000 hPa的涡度数据
            vo_data = ds['vo'].sel({level_dim: 850})

            # 使用进程池并限制最大进程数
            with ProcessPoolExecutor(max_workers=16) as executor:
                time_chunks = chunk_list(ds[time_dim].values, chunk_size=5)
                futures = [
                    executor.submit(process_time_step_chunk, chunk, vo_data, iParams, time_dim, area)
                    for chunk in time_chunks
                ]
                # 使用tqdm显示该小时的进度
                all_sf.extend([sf for future in futures for sf in future.result()])

    # 如果该月有数据，合并并保存
    if all_sf:
        sf_data = xr.concat(all_sf, dim=time_dim).sortby(time_dim)
        sf_data.name = 'streamfunction'
        sf_data.attrs = {
            'units': 'm^2 s^-1',
            'long_name': f'Streamfunction at 850 hPa'
        }

        ds_out = xr.Dataset({'streamfunction': sf_data})
        ds_out.attrs = {
            'Conventions': 'CF-1.7',
            'source': f'Calculated from ERA5 vo at 850 hPa using xinvert',
            'history': f'Created on {np.datetime_as_string(np.datetime64("now"), unit="s")}'
        }

        ds_out.to_netcdf(out_path)
    else:
        print(f"⚠️ {year}-{month:02d} 没有数据，跳过")

if __name__ == '__main__':
    # 主循环，使用tqdm显示年月进度
    year_month = [(year, month) for year in range(2023, 2024) for month in range(1, 13)]
    for year, month in tqdm(year_month, desc="处理年月"):
        process_month(year, month, hour)