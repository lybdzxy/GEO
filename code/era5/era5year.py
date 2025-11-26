import xarray as xr
from datetime import datetime, timedelta
import pandas as pd

# 创建一个空的xarray数据集来存储提取的数据
era5_combined = xr.Dataset()

for year in range(2019, 2023, 1):
    coryear = year + 1
    input_path = f'E:/GEO/ERA5/cor/cor{year}_1_1.nc'
    era5 = xr.open_dataset(input_path)
    firstday = datetime(year, 1, 2)
    lastday = datetime(coryear, 1, 1)
    era5cor = era5.sel(time=slice(firstday, lastday))
    # 将时间变量转换为 Pandas Series
    time_series = era5cor['time'].to_series()

    # 前移一天
    time_series = time_series - timedelta(days=1)

    # 更新xarray数据集的时间变量
    era5cor['time'] = time_series.values

    output_path = f'E:/GEO/ERA5/cor/cor{year}.nc'
    era5cor.to_netcdf(output_path)
