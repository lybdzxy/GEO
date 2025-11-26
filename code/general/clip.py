import xarray as xr
import geopandas as gpd
import salem

shp_dir = f'E:/GEO/geodata/CC.shp'
ming_shp2 = gpd.read_file(shp_dir)

# 输入文件路径
nc_path = f'E:/GEO/test/obs.nc'
output_nc_path = f'E:/GEO/test/obs-cc.nc'
ds = xr.open_dataset(nc_path)

# 对空间维度进行裁剪
ming = ds.salem.roi(shape=ming_shp2)

# 获取要计算平均值的变量
variable_names = ming.variables.keys()

# 检查 '__xarray_dataarray_variable__' 是否在列表中
if '__xarray_dataarray_variable__' in variable_names:
    variable_name = '__xarray_dataarray_variable__'
elif 'pre' in variable_names:
    variable_name = 'pre'

ming['pre'] = ming[variable_name].astype('float64')  # 确保 'pr' 存储为双精度浮点数
ming.to_netcdf(output_nc_path)
