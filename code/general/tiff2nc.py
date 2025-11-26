import os
from osgeo import gdal
import netCDF4 as nc
import numpy as np

# 输入TIFF文件路径和名称
tiff_file = 'E:/benji/example/mao_2005_01.tif.tif'

# 设置输出NetCDF文件名称
output_nc_file = 'output.nc'

# 打开TIFF文件
tiff_dataset = gdal.Open(tiff_file)

# 获取TIFF文件的地理信息
geo_transform = tiff_dataset.GetGeoTransform()
start_lon = geo_transform[0]
start_lat = geo_transform[3]
lon_resolution = geo_transform[1]
lat_resolution = geo_transform[5]
width = tiff_dataset.RasterXSize
height = tiff_dataset.RasterYSize
end_lon = start_lon + (width * lon_resolution)
end_lat = start_lat + (height * lat_resolution)

# 创建NetCDF文件
ncfile = nc.Dataset(output_nc_file, 'w', format='NETCDF4')

# 创建维度
x_dim = ncfile.createDimension('x', width)
y_dim = ncfile.createDimension('y', height)

# 创建经度和纬度变量
lon = ncfile.createVariable('lon', 'f4', ('x',))
lat = ncfile.createVariable('lat', 'f4', ('y',))

# 添加坐标数据
lon[:] = np.linspace(start_lon, end_lon, width)
lat[:] = np.linspace(start_lat, end_lat, height)

# 添加地理元数据
ncfile.lon_min = start_lon
ncfile.lon_max = end_lon
ncfile.lat_min = start_lat
ncfile.lat_max = end_lat
ncfile.projection = tiff_dataset.GetProjection()

# 添加数据变量
data = ncfile.createVariable('data', 'f4', ('y', 'x'))

# 从TIFF文件中读取数据并将其存储到NetCDF文件中
data_array = tiff_dataset.ReadAsArray()
data[:] = data_array

# 关闭文件
ncfile.close()
tiff_dataset = None

