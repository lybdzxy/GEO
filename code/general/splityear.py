import os
import xarray as xr
import dask
import numpy as np
from tqdm import tqdm

# 文件夹路径
folder_path = 'E:/GEO/CMIP6/fulltime/'

# 获取文件夹下所有的 NetCDF 文件
nc_files = [file for file in os.listdir(folder_path) if file.endswith('.nc')]

# 使用 tqdm 追踪进度
for file_name in tqdm(nc_files, desc='Processing files'):
    # 构建文件路径
    file_path = os.path.join(folder_path, file_name)

    # 加载 NetCDF 文件
    ds = xr.open_dataset(file_path)

    # 使用 dask 进行并行计算
    ds = ds.chunk({'time': 'auto'})


    # 定义一个函数，用于拆分并保存数据
    def process_and_save(year):
        yearly_data = ds.sel(time=ds.time.dt.year == year)
        output_file_path = os.path.join(folder_path, f'{os.path.splitext(file_name)[0]}_{year}.nc')
        yearly_data.to_netcdf(output_file_path)
        print(f'Year {year} data saved to {output_file_path}')


    # 使用 dask 进行并行处理
    years = np.unique(ds.time.dt.year.values)
    tasks = [dask.delayed(process_and_save)(year) for year in years]

    # 执行并行任务
    dask.compute(*tasks)

    # 关闭数据集
    ds.close()

print("All files processed successfully.")
