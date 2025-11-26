import os
import xarray as xr

# 指定坐标文件路径
coor_path = 'E:/GEO/test/CHM_PRE_0.5dg_19612022.nc'
coor = xr.open_dataset(coor_path)

# 指定待处理文件夹路径
folder_path = 'E:/GEO/CMIP6/ec3/'

# 列出文件夹下的所有文件
all_files = os.listdir(folder_path)

# 遍历文件夹下的每个文件
for file_name in all_files:
    # 构建文件的完整路径
    file_path = os.path.join(folder_path, file_name)

    # 检查文件是否为NetCDF文件
    if file_name.endswith('.nc'):
        # 读取NetCDF文件
        ds = xr.open_dataset(file_path)
        pr = ds['pr']

        # 定义插值网格
        new_lon = coor['longitude']  # 新的经度点数
        new_lat = coor['latitude']  # 新的纬度点数

        # 进行双线性插值
        interp_pr = pr.interp(lat=new_lat, lon=new_lon, method='linear')

        # 关闭原始数据集
        ds.close()

        # 保存裁剪后的数据为新的NetCDF文件
        output_file_path = os.path.join(folder_path, f'interpolated_{file_name}')
        interp_pr.to_netcdf(output_file_path)
