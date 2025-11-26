import xarray as xr
import numpy as np
import pymannkendall as mk
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

ssp_values = [126, 245,370,585]  # 将字符串改为整数
index_values = ['prcptot','r95p','r99p','r95ptot','r99ptot','sdii','rx1day','rx5day','r10mm','r20mm','cwd']  # 保持字符串不变
years = range(2015, 2080)
lons = np.arange(72.45, 137.45, 1)
lats = np.arange(54.55, 17.55, -1)


def perform_mk_test(ssp, index, lon, lat):
    totest = np.array([])  # 用于存储数据的数组
    for year in years:
        data_path = f'E:/GEO/etccdi/{ssp}/{index}/cut10x10{index}{year}.nc'
        data = xr.open_dataset(data_path)['pr'].sel(longitude=lon, latitude=lat)
        totest = np.append(totest, data)
    try:
        mk_result = mk.original_test(totest)
        trend, _, p, z, Tau, s, var_s, slope, intercept = mk_result
        # 转换trend为0, 1, -1
        trend_value = 0 if trend == 'no trend' else 1 if trend == 'increasing' else -1
        return (lon, lat, trend_value, p, z, Tau, s, var_s, slope, intercept)
    except ZeroDivisionError:
        return None


if __name__ == '__main__':
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

            # 提取 MK 检验结果并保存为 Pandas 数据帧
            mk_results_df = pd.DataFrame(result).T
            if mk_results_df.empty:
                print(ssp, index, "没有收集到结果")
                continue  # 跳过当前迭代

            # 如果不为空，分配列名
            mk_results_df.columns = ['trend', 'p', 'z', 'Tau', 's', 'var_s', 'slope', 'intercept']

            # 将数据帧保存为 NetCDF 文件
            mk_results_df.to_xarray().to_netcdf(f'E:/GEO/{ssp}{index}_mktest.nc')