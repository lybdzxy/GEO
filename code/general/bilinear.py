import xarray as xr
import geopandas as gpd
import salem

shp_dir = 'E:/GEO/result/geodata/ecm_el.shp'
shp = gpd.read_file(shp_dir)
# 指定坐标文件路径
coor_path = 'E:/GEO/etccdi/r99p2015.nc'
coor = xr.open_dataset(coor_path)

# 指定待处理文件夹路径
file_path = 'E:/GEO/result/geodata/dem.nc'

ds = xr.open_dataset(file_path)
pr = ds['dem']
coor = coor.rename({'longitude': 'lon', 'latitude': 'lat'})
# 定义插值网格
new_lon = coor['lon']  # 新的经度点数
new_lat = coor['lat']  # 新的纬度点数

# 进行双线性插值
interp_pr = pr.interp(lat=new_lat, lon=new_lon, method='linear')
interp_pr = interp_pr.salem.roi(shape=shp)

# 关闭原始数据集
ds.close()

# 保存裁剪后的数据为新的NetCDF文件
output_file_path = 'E:/GEO/result/geodata/dem_resample_cut.nc'
interp_pr.to_netcdf(output_file_path)
