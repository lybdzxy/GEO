import xarray as xr


# 定义一个函数来合并指定年份的数据
def merge_years(year):
    # 读取年份的数据
    data_year = xr.open_dataset(f'E:/GEO/ERA5/{year}.nc')

    # 读取年初的数据
    data_year_start = xr.open_dataset(f'E:/GEO/ERA5/{year}_1_1.nc')

    # 合并这两个数据集
    merged_data = xr.concat([data_year, data_year_start], dim="time")

    # 将合并后的数据保存为新的文件
    merged_data.to_netcdf(f'cor{year}_1_1.nc')


# 合并从1950年到2018年的数据
for year in range(2019, 2023):
    merge_years(year)
