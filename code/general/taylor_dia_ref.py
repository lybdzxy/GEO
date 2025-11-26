import numpy as np
import xarray as xr
import pandas as pd

def calculate_standard_deviation(data):
    std_dev = np.std(data)
    return std_dev

# 创建空的 DataFrame 以保存指标
results_df = pd.DataFrame(columns=['Index', 'Std_Dev_Predicted'])

indices = ['prcptot', 'r95p', 'r99p', 'r95ptot', 'r99ptot', 'sdii', 'rx1day', 'r10mm', 'r20mm']

for index in indices:
    std = []  # 修复：在每次循环迭代前清空 std 数组
    for year in range(1988, 2015):
        # 文件路径
        file_predicted = f'E:/GEO/etccdi/chm_pre/cut/{index}{year}.nc'

        # 打开 NetCDF 文件
        dataset_predicted = xr.open_dataset(file_predicted)

        # 读取变量数据为数组
        predicted = dataset_predicted['pre'].values

        # 去除NaN值
        predicted = predicted[~np.isnan(predicted)]

        # 计算指标
        std_dev_predicted = calculate_standard_deviation(predicted)
        std.append(std_dev_predicted)  # 修复：将每次迭代的标准差值添加到 std 数组中
    std_mean = np.mean(std)
    # 添加到 DataFrame
    results_df = results_df.append({'Index': index, 'Std_Dev_Predicted': std_mean}, ignore_index=True)

# 保存结果到 Excel 文件
results_df.to_excel('E:/GEO/etccdi/qpm/ref.xlsx', index=False)
