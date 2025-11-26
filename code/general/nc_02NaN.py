import xarray as xr
import numpy as np

# 打开数据集
data_path = 'E:/GEO/result/geodata/3000m_dem.nc'
data = xr.open_dataset(data_path)

# 将数据转换为 NumPy 数组并替换值
data_np = data['dem'].values
data_np = np.where(data_np == 0, np.nan, data_np)

# 将替换后的数组重新包装成 DataArray 对象
data['dem'] = xr.DataArray(data_np, coords=data['dem'].coords, dims=data['dem'].dims)

# 保存到新的 NetCDF 文件
output_path = 'E:/GEO/result/geodata/3000m_dem_nan.nc'
data.to_netcdf(output_path)
