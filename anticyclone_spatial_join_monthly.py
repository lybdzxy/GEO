import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime

# 读取 CSV 文件
poi_data = pd.read_csv("trajectories_filtered.csv")

# 创建 360 x 90 网格
lon_bins = np.arange(-180, 181, 1)  # 经度范围 -180° 到 180°（包含 180°）
lat_bins = np.arange(0, 91, 1)      # 纬度范围 0° 到 90°（北半球）

# 提取年月（格式为 'YYYY-MM'）
poi_data['month'] = poi_data['date'].apply(lambda x: datetime.strptime(str(x), "%Y%m%d%H").strftime("%Y-%m"))

# 调整 POI 经度为标准的 -180 到 180 范围
poi_data['center_lon'] = np.where(poi_data['center_lon'] > 180, poi_data['center_lon'] - 360, poi_data['center_lon'])

# 获取所有唯一的年月，按时间顺序排序
months = sorted(poi_data['month'].unique())

# 初始化 NetCDF 文件
dataset = nc.Dataset('traj.nc', 'w', format='NETCDF4')

# 创建维度
dataset.createDimension('lon', len(lon_bins) - 1)
dataset.createDimension('lat', len(lat_bins) - 1)
dataset.createDimension('month', len(months))

# 创建经纬度和月份的变量
lon_var = dataset.createVariable('lon', np.float32, ('lon',))
lat_var = dataset.createVariable('lat', np.float32, ('lat',))
month_var = dataset.createVariable('month', str, ('month',))

# 给经纬度和月份变量赋值
lon_var[:] = lon_bins[:-1] + 0.5  # 栅格中心点
lat_var[:] = lat_bins[:-1] + 0.5
month_var[:] = np.array(months, dtype='S7')  # 字符串形式的年月

# 创建 POI 数量变量
poi_count = dataset.createVariable('traj', np.int32, ('month', 'lat', 'lon'))

# 初始化 POI 计数器
poi_count_data = np.zeros((len(months), len(lat_bins) - 1, len(lon_bins) - 1), dtype=int)

# 统计每个月每个格点的 POI 数量
for _, row in poi_data.iterrows():
    lat_idx = np.digitize(row['center_lat'], lat_bins) - 1
    lon_idx = np.digitize(row['center_lon'], lon_bins) - 1
    month_idx = months.index(row['month'])

    # 保证索引不越界
    lat_idx = min(max(lat_idx, 0), len(lat_bins) - 2)
    lon_idx = min(max(lon_idx, 0), len(lon_bins) - 2)

    poi_count_data[month_idx, lat_idx, lon_idx] += 1

# 写入数据到 NetCDF 文件
poi_count[:] = poi_count_data

# 关闭文件
dataset.close()

print("NetCDF 文件已生成：traj.nc")
