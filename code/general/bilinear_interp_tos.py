import os
import xarray as xr

# 指定坐标文件路径
coor_path = 'E:/GEO/CMIP6/tos/fulltime/tos_Omon_GFDL-ESM4_ssp126.nc'

try:
    coor = xr.open_dataset(coor_path)
except FileNotFoundError:
    print(f"坐标文件 {coor_path} 未找到。")
    raise

# 指定待处理文件夹路径
folder_path = 'E:/GEO/CMIP6/tos/fulltime/'

if not os.path.isdir(folder_path):
    print(f"文件夹 {folder_path} 不存在。")
    raise FileNotFoundError

# 列出文件夹下的所有文件
all_files = os.listdir(folder_path)

# 定义插值网格
new_lon = coor['x'].values  # 新的经度点数
new_lat = coor['y'].values  # 新的纬度点数

# 遍历文件夹下的每个文件
for file_name in all_files:
    # 构建文件的完整路径
    file_path = os.path.join(folder_path, file_name)
    # 检查文件是否为NetCDF文件
    if file_name.endswith('.nc'):
        with xr.open_dataset(file_path) as ds:
            # 尝试找到实际使用的维度名称
            lat_dim = None
            lon_dim = None

            for dim in ds.dims:
                if 'lat' in dim:
                    lat_dim = dim
                elif 'lon' in dim:
                    lon_dim = dim
                elif 'j' in dim:
                    lat_dim = dim
                elif 'i' in dim:
                    lon_dim = dim
                elif 'y' in dim:
                    lat_dim = dim
                elif 'x' in dim:
                    lon_dim = dim

            if not lat_dim or not lon_dim:
                print(f"文件 {file_path} 中未找到纬度或经度维度。")
                continue

            pr = ds['tos']

            # 进行时间切片，选取2015到2068年的数据
            pr = pr.sel(time=slice('2015-01-01', '2068-12-31'))

            # 计算年平均值
            pr_annual_mean = pr.resample(time='Y').mean()

            # 进行双线性插值
            interp_pr = pr_annual_mean.interp({lat_dim: new_lat, lon_dim: new_lon}, method='linear')

            # 保存裁剪后的数据为新的NetCDF文件
            output_file_path = os.path.join(folder_path, f'interpolated_annual_mean_{file_name}')
            interp_pr.to_netcdf(output_file_path)
            print(f"插值并计算年平均完成，保存到 {output_file_path}")


print("所有文件处理完成。")