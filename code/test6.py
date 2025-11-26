import xarray as xr
from xinvert import invert_Poisson
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings
from contextlib import redirect_stdout
import multiprocessing

warnings.filterwarnings("ignore")

hour = [0, 6, 12, 18]

def chunk_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def worker_process_time_chunk(data_path, times, level_dim_name, time_dim_name, iParams):
    """
    在子进程中打开 dataset，处理一组时间步 —— 避免将大 DataArray 传给子进程导致 pickle 开销。
    返回：list of DataArray (每个含 time 维)
    """
    results = []
    if not os.path.exists(data_path):
        # 返回空列表，主进程会忽略
        return results

    try:
        with xr.open_dataset(data_path) as ds:
            ds = ds.sortby('latitude')
            lat = ds['latitude']
            lon = ds['longitude']

            # 计算面积权重（2D）
            area = np.cos(np.deg2rad(lat))
            # expand_dims 用 xarray 方式以保证索引对齐
            area = xr.DataArray(area, coords={'latitude': lat}, dims=['latitude'])
            area = area.expand_dims(longitude=lon)  # dims: [latitude, longitude]

            vo = ds['vo'].sel({level_dim_name: 850})  # 选 850 hPa（与你原脚本一致）

            for t in times:
                # 有时 t 可能为 numpy.datetime64；使用 .sel 按索引选取
                try:
                    with open(os.devnull, 'w') as fnull:
                        with redirect_stdout(fnull):
                            vo_t = vo.sel({time_dim_name: t})
                            # 修正涡度，使全球积分为零
                            bias = (vo_t * area).sum() / area.sum()
                            vo_t = vo_t - bias
                            sf = invert_Poisson(vo_t, dims=['latitude', 'longitude'], iParams=iParams)
                            sf = sf.astype('float32')
                            # 保证有 time 维以便后续 concat
                            sf = sf.expand_dims({time_dim_name: [t]})
                            results.append(sf)
                            # 及时删除局部变量
                            del vo_t, sf
                except Exception:
                    # 若单步出错，跳过该步（主进程可记录日志）
                    continue
    except Exception:
        # 若打开文件或整体处理失败，直接返回空列表（可在主进程记录）
        return results

    return results

def process_month(year, month, hour, executor, max_chunk_size=30):
    """
    使用传入的 executor（常驻进程池）提交子任务并异步收集结果。
    max_chunk_size：每个子任务处理的时间步数量（可调，默认 30）
    """
    out_path = f"F:/ERA5/hourly/lvl/stream/ERA5_stream_850hpa_{year}{month:02d}.nc"
    if os.path.exists(out_path):
        print(f"已存在: {out_path}，跳过")
        return

    all_futures = []
    time_dim_global = None
    level_dim_global = None

    # 反演参数（保持原设置）
    iParams = {
        'BCs': ['extend', 'periodic'],
        'mxLoop': 1000,
        'tolerance': 1e-6,
    }

    # 为了尽量减少打开文件次数：每个时次（0z/6z/12z/18z）对应一个 data_path
    for h in hour:
        data_path = f'F:/ERA5/hourly/lvl/{h}z/ERA5_{h}z_lvl_{year}{month:02d}.nc'
        if not os.path.exists(data_path):
            print(f"缺少文件: {data_path}")
            continue

        # 先快速打开一次以获得 time_dim 名称（不读取完整数据）
        try:
            with xr.open_dataset(data_path) as ds:
                ds = ds.sortby('latitude')
                time_dim = 'time' if 'time' in ds.dims else 'valid_time'
                level_dim = 'level' if 'level' in ds.dims else 'pressure_level'
                times = list(ds[time_dim].values)
                # 记录以便最终 concat 使用一致的 time_dim 名称
                time_dim_global = time_dim_global or time_dim
                level_dim_global = level_dim_global or level_dim
        except Exception:
            print(f"无法打开或读取: {data_path}")
            continue

        # 切分时间步以增大任务粒度（每个子任务处理较多时间步）
        chunk_size = max_chunk_size
        time_chunks = chunk_list(times, chunk_size=chunk_size)

        for chunk in time_chunks:
            # 提交任务：只传入路径和时间列表（避免传大 DataArray）
            fut = executor.submit(worker_process_time_chunk, data_path, chunk, level_dim, time_dim, iParams)
            all_futures.append((fut, data_path))  # 记录 data_path 便于错误定位

    # 异步收集结果并合并
    all_sf = []
    if all_futures:
        futures_only = [f for f, _ in all_futures]
        for fut in as_completed(futures_only):
            try:
                res = fut.result()
                if res:
                    all_sf.extend(res)
            except Exception as e:
                # 打印出错的 future（可拓展为写入日志）
                print("子任务异常：", e)
                continue

    # 合并并保存（若有数据）
    if all_sf:
        # concat 之前确保 time_dim 名称存在（使用 first 的 time_dim 名）
        concat_dim = time_dim_global or 'time'
        sf_data = xr.concat(all_sf, dim=concat_dim).sortby(concat_dim)
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

        # 写出 nc 文件（阻塞）
        ds_out.to_netcdf(out_path)
    else:
        print(f"⚠️ {year}-{month:02d} 没有数据，跳过")

if __name__ == '__main__':
    # 全局常驻进程池：只创建一次
    cpu_count = multiprocessing.cpu_count()
    max_workers = 16 # 留一个核给系统
    print(f"检测到 CPU 核数: {cpu_count}, 使用 max_workers = {max_workers}")

    # 可调参数：每个子任务处理的时间步数（增大可减少调度开销）
    MAX_CHUNK_SIZE = 40

    # 年月队列
    year_month = [(year, month) for year in range(1940, 2024) for month in range(1, 13)]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 主循环：只负责提交任务与收集结果（executor 常驻）
        for year, month in tqdm(year_month, desc="处理年月"):
            process_month(year, month, hour, executor, max_chunk_size=MAX_CHUNK_SIZE)
