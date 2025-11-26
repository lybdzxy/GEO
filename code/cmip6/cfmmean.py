import xarray as xr

addredailymean = {}

# 指定文件名模式
file_pattern = 'E:/GEO/CMIP6/addredaily/addredaily{model}_{start}_{end}.nc'

# 遍历时间段
for start in range(1950, 2011, 5):
    end = start + 4  # 计算5年期间的结束年份

    # 使用延迟加载打开和加载数据，包括在时间段内的所有模型
    data = xr.open_mfdataset(
        [file_pattern.format(model=model, start=start, end=end) for model in range(1, 11)],
        combine='nested',  # 使用'by_coords'参数时，需要同时指定'concat_dim'，此处使用'nested'
        concat_dim='model'  # 添加'model'维度
    )

    # 计算均值并将其存储在字典中
    addredailymean[start] = data['pr'].mean(dim='model')

    # 定义均值数据的输出路径
    output_path = f'E:/GEO/CMIP6/addredailymean/addredailymean{start}_{end}.nc'

    # 将均值数据保存到NetCDF文件中
    addredailymean[start].to_netcdf(output_path)
