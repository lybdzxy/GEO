import xarray as xr
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import tqdm
import numpy as np
from scipy.stats import genextreme
from lmoments3 import distr

ssp_values = [126, 585]  # 将字符串改为整数
index_values = ['r95p', 'r99p', 'rx1day']  # 保持字符串不变
years = range(2015, 2069)
lons = np.arange(72.25, 136.25, 0.5)
lats = np.arange(53.75, 17.75, -0.5)


def perform_gev(ssp, index, lon, lat):
    totest = np.array([])  # 用于存储数据的数组
    for year in years:
        data_path = f'E:/GEO/etccdi/qpm/mme/new/{index}_{ssp}_{year}.nc'
        data = xr.open_dataset(data_path)['__xarray_dataarray_variable__'].sel(lon=lon, lat=lat)
        totest = np.append(totest, data)

    if np.all(totest == 0):
        return None
    try:
        paras = distr.gev.lmom_fit(totest)

        loc = paras['loc']
        scale = paras['scale']
        shape = paras['c']

        # 计算不同重现期对应的降水量
        precipitation_3_year = genextreme.ppf(1 - 1 / 3, shape, loc=loc, scale=scale)
        precipitation_10_year = genextreme.ppf(1 - 1 / 10, shape, loc=loc, scale=scale)
        precipitation_20_year = genextreme.ppf(1 - 1 / 20, shape, loc=loc, scale=scale)
        precipitation_50_year = genextreme.ppf(1 - 1 / 50, shape, loc=loc, scale=scale)
        precipitation_100_year = genextreme.ppf(1 - 1 / 100, shape, loc=loc, scale=scale)

        return (lon, lat, loc, scale, shape, precipitation_3_year, precipitation_10_year, precipitation_20_year, precipitation_50_year, precipitation_100_year)
    except (ZeroDivisionError, ValueError, RuntimeError):
        return None

if __name__ == '__main__':
    total_tasks = 6 * len(lons) *len(lats)
    with tqdm.tqdm(total=total_tasks, desc="处理中") as pbar:
        for ssp in ssp_values:
            for index in index_values:
                result = {}  # 用于存储MK检验结果的字典
                tasks = []

                with ProcessPoolExecutor(max_workers=12) as executor:  # 可以根据需要调整max_workers
                    for lon in lons:
                        for lat in lats:
                            tasks.append(executor.submit(perform_gev, ssp, index, lon, lat))

                for task in tasks:
                    if task.result() is not None:
                        lon, lat, loc, scale, shape, precipitation_3_year, precipitation_10_year, precipitation_20_year, precipitation_50_year, precipitation_100_year = task.result()
                        result[(lon, lat)] = (loc, scale, shape, precipitation_3_year, precipitation_10_year, precipitation_20_year, precipitation_50_year, precipitation_100_year)

                # 提取 MK 检验结果并保存为 Pandas 数据帧
                gev_results_df = pd.DataFrame(result).T
                if gev_results_df.empty:
                    print(ssp, index, "没有收集到结果")
                    continue  # 跳过当前迭代

                # 如果不为空，分配列名
                gev_results_df.columns = ['loc', 'scale', 'shape', 'precipitation_3_year', 'precipitation_10_year', 'precipitation_20_year', 'precipitation_50_year', 'precipitation_100_year']

                # 将数据帧保存为 NetCDF 文件
                gev_results_df.to_xarray().to_netcdf(f'E:/GEO/result/new/{ssp}{index}_lom_GEV.nc')
                for task in concurrent.futures.as_completed(tasks):
                    task.result()  # 等待任务完成，不处理返回值
                    pbar.update(1)
