import xarray as xr
import numpy as np

# 打开数据集
data_path = 'E:/GEO/chm_pre/CHM_PRE_0.1dg_19612022/CHM_PRE_0.1dg_19612022.nc'
data = xr.open_dataset(data_path)

# 将负值替换为 NaN
data['pre'] = data['pre'].where(data['pre'] >= 0)

# 计算每年的总降水量
annual_precip = data['pre'].groupby('time.year').sum(dim='time')

# 计算多年平均年降水量
mean_annual_precip = annual_precip.mean(dim='year')

# 提取400mm年等降水量线
precip_400mm_mask = mean_annual_precip >= 400

# 创建新的数据集
precip_400mm_ds = xr.Dataset(
    {
        'precip_400mm_mask': (['latitude', 'longitude'], precip_400mm_mask.data)
    },
    coords={
        'latitude': mean_annual_precip['latitude'],
        'longitude': mean_annual_precip['longitude']
    }
)

# 保存到新的 NetCDF 文件
output_path = 'E:/GEO/result/geodata/400mm_annual_precipitation.nc'
precip_400mm_ds.to_netcdf(output_path)

print(f"400mm annual precipitation mask has been saved to {output_path}")
