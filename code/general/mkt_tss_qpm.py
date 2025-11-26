import xarray as xr
import numpy as np
import pymannkendall as mk
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import tqdm

ssp_values = [126,585]  # 将字符串改为整数
index_values = ['prcptot','r95p','r99p','sdii','rx1day']  # 保持字符串不变
years = range(2015, 2069)
lons = np.arange(72.25, 136.25, 0.5)
lats = np.arange(53.75, 17.75, -0.5)


def perform_mk_test(ssp, index, lon, lat):
    totest = np.array([])  # 用于存储数据的数组
    for year in years:
        data_path = f'E:/GEO/etccdi/qpm/mme/ecm/{index}_{ssp}_{year}.nc'
        dataset = xr.open_dataset(data_path)
        data = dataset['__xarray_dataarray_variable__'].sel(lon=lon, lat=lat)
        totest = np.append(totest, data)

    if totest.size == 0:  # 检查数据点数量
        return (lon, lat, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    try:
        mk_result = mk.original_test(totest)
        trend, _, p, z, Tau, s, var_s, slope, intercept = mk_result
        trend_value = 0 if trend == 'no trend' else 1 if trend == 'increasing' else -1
        return (lon, lat, trend_value, p, z, Tau, s, var_s, slope, intercept)
    except ZeroDivisionError:
        return (lon, lat, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)


if __name__ == '__main__':
    total_tasks = 2 * 5 *len(lons)*len(lats)
    with tqdm.tqdm(total=total_tasks, desc="处理中") as pbar:
        for ssp in ssp_values:
            for index in index_values:
                result = {}  # 用于存储MK检验结果的字典
                tasks = []

                with ProcessPoolExecutor(max_workers=12) as executor:  # 可以根据需要调整max_workers
                    for lon in lons:
                        for lat in lats:
                            tasks.append(executor.submit(perform_mk_test, ssp, index, lon, lat))

                for task in tasks:
                    if task.result() is not None:
                        lon, lat, trend, p, z, Tau, s, var_s, slope, intercept = task.result()
                        result[(lon, lat)] = (trend, p, z, Tau, s, var_s, slope, intercept)

                for task in concurrent.futures.as_completed(tasks):
                    task.result()  # 等待任务完成，不处理返回值
                    pbar.update(1)

                # 提取 MK 检验结果并保存为 Pandas 数据帧
                mk_results_df = pd.DataFrame(result).T
                if mk_results_df.empty:
                    print(ssp, index, "没有收集到结果")
                    continue  # 跳过当前迭代

                # 如果不为空，分配列名
                mk_results_df.columns = ['trend', 'p', 'z', 'Tau', 's', 'var_s', 'slope', 'intercept']

                # 将数据帧保存为 NetCDF 文件
                data_path = f'E:/GEO/etccdi/qpm/mme/ecm/{index}_{ssp}_2025.nc'
                dataset = xr.open_dataset(data_path)
                mk_result_xr = mk_results_df.to_xarray()
                mk_result_xr = mk_result_xr.assign_coords(lon=dataset.lon, lat=dataset.lat)
                mk_result_xr.to_netcdf(f'E:/GEO/result/ecm/{ssp}{index}_mktest.nc')