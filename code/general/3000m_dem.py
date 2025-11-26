import xarray as xr
import numpy as np

# 打开数据集
data_path = 'E:/GEO/result/geodata/dem.nc'
data = xr.open_dataset(data_path)
print(data)
# 提取400mm年等降水量线
dem_3000 = data['dem'] <= 3000

# 创建新的数据集
dem_3000_ds = xr.Dataset(
    {
        'dem': (['lat', 'lon'], dem_3000.data)
    },
    coords={
        'latitude': data['dem']['lat'],
        'longitude': data['dem']['lon']
    }
)

# 保存到新的 NetCDF 文件
output_path = 'E:/GEO/result/geodata/3000m_dem.nc'
dem_3000_ds.to_netcdf(output_path)

print(f"3000 dem mask has been saved to {output_path}")
