import xarray as xr
from datetime import datetime

# 创建一个空的xarray数据集来存储提取的数据
era5_combined = xr.Dataset()

for year in range(2020, 2024, 1):
    coryear = year - 1
    input_path = f'E:/GEO/ERA5/{year}.nc'
    era5 = xr.open_dataset(input_path)
    date = datetime(year, 1, 1)
    era5_fird = era5.sel(time=date)

    # 添加时间坐标到数据变量中
    era5_fird['time'] = [date]

    # 将每年的数据添加到组合数据集中
    era5_combined[f'ERA5_{coryear}'] = era5_fird.to_array()

    output_path = f'E:/GEO/ERA5/{coryear}_1_1.nc'
    era5_fird.to_netcdf(output_path)
