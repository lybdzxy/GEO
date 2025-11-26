import netCDF4 as nc
import pandas as pd
import openpyxl
import numpy as np

# 创建一个空的数据框来存储结果
ssp_values=('his')
indices = ('prcptot','r95p','r99p','r95ptot','r99ptot','sdii','rx1day','rx5day','r10mm','r20mm','cwd')
# 循环处理每个文件
#for ssp in ssp_values:
for index in indices:
    result_df = pd.DataFrame(columns=['Year', 'Mean Value'])
    for year in range(1950, 2023):
        # 打开当前文件
        file_name = f'E:/GEO/etccdi/his/{index}/cut{index}{year}.nc'
        nc_file = nc.Dataset(file_name)

        # 获取要计算平均值的变量
        variable_names = nc_file.variables.keys()

        # 检查 '__xarray_dataarray_variable__' 是否在列表中
        if '__xarray_dataarray_variable__' in variable_names:
            variable_name = '__xarray_dataarray_variable__'
        elif 'tp' in variable_names:
            variable_name = 'tp'

        variable = nc_file.variables[variable_name]

        # 计算年度平均值
        yearly_mean = np.nanmean(variable)  # 假设时间维度为 'time'

        # 将结果添加到数据框
        result_df = result_df.append({'Year': year, 'Mean Value': yearly_mean}, ignore_index=True)

        # 关闭当前文件
        nc_file.close()

    # 将结果保存为Excel文件
    excel_file_name = f'E:/GEO/etccdi/his/{index}.xlsx'  # 指定Excel文件名
    result_df.to_excel(excel_file_name, index=False)

    print(f'年度平均值已保存到 {excel_file_name}')

