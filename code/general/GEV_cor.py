import xarray as xr
import numpy as np
from scipy.stats import zscore
import tqdm
import concurrent.futures
import pandas as pd

lons = np.arange(72.25, 136.25, 0.5)
lats = np.arange(53.75, 17.75, -0.5)

def perform_zs(ssp, index, name, lon, lat):
    data_path = f'E:/GEO/result/qpm/{ssp}{index}_cor2_GEV.nc'
    dataset = xr.open_dataset(data_path)
    data = dataset[name]
    totest = np.array([])

    for la in (1, 0.5, 0, -0.5, -1):
        for lo in (-1, -0.5, 0, 0.5, 1):
            new_lon = lon + lo
            new_lat = lat + la
            if new_lon not in data.level_0.values or new_lat not in data.level_1.values:
                beside = np.nan
            else:
                beside = data.sel(level_0=new_lon, level_1=new_lat).values
            totest = np.append(totest, beside)
    nan_count = np.sum(np.isnan(totest[:12]))
    center = 12-nan_count
    valid_values = totest[~np.isnan(totest)]
    zs = zscore(valid_values)
    if totest[12] in valid_values and not np.isnan(totest[12]) and zs[center] > 2:
        pre = np.nan
    else:
        pre = totest[12] if not np.isnan(totest[12]) else np.nan
    return (lon, lat, pre)

def main():
    names = ['precipitation_3_year', 'precipitation_10_year', 'precipitation_20_year', 'precipitation_50_year', 'precipitation_100_year']
    indices = ['r95p', 'r99p', 'rx1day']
    ssps = ['126', '245', '370', '585']
    total_tasks = len(ssps) * len(indices) * len(names) * len(lons) * len(lats)
    with tqdm.tqdm(total=total_tasks, desc="处理中") as pbar:
        for ssp in ssps:
            for index in indices:
                result = {}  # 用于存储zs检验结果的字典
                tasks = []
                for name in names:
                    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:  # 可以根据需要调整max_workers
                        for lon in lons:
                            for lat in lats:
                                tasks.append(executor.submit(perform_zs, ssp, index, name, lon, lat))
                                pbar.update(1)

                    for task in tasks:
                        if task.result() is not None:
                            lon, lat, pre = task.result()
                            result[(lon, lat)] = pre

                    # 提取 zs 检验结果并保存为 Pandas 数据帧
                    zs_results_df = pd.DataFrame(result.values(), index=result.keys(), columns=['pre'])
                    if zs_results_df.empty:
                        print(index, "没有收集到结果")
                        continue  # 跳过当前迭代

                    # 将数据帧保存为 NetCDF 文件
                    zs_results_df.to_xarray().to_netcdf(f'E:/GEO/result/qpm/gev/{ssp}{index}_{name}.nc')

if __name__ == '__main__':
    main()