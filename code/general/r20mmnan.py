import netCDF4 as nc
import numpy as np

for year in range(2015, 2080):
    # 输入文件路径
    nc_path = f'E:/GEO/etccdi/126/r20mm/cutr20mm{year}.nc'

    # 使用上下文管理器打开NetCDF文件
    with nc.Dataset(nc_path, 'r+') as nc_file:
        variable_names = nc_file.variables.keys()

        # 检查 '__xarray_dataarray_variable__' 是否在列表中
        if '__xarray_dataarray_variable__' in variable_names:
            variable_name = '__xarray_dataarray_variable__'
        elif 'pr' in variable_names:
            variable_name = 'pr'

        # 获取要处理的变量（假设变量名为'data'）
        data_var = nc_file.variables[variable_name]

        # 提取变量的数据
        data = data_var[:]

        # 将数据类型转换为浮点数
        data = data.astype(float)

        # 将小于0的值设为NaN
        data[data < 0] = np.nan

        # 将修改后的数据写回变量
        data_var[:] = data

        # 文件将在上下文管理器退出时自动关闭
