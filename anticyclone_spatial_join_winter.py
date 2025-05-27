import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime

# 读取 CSV 文件
poi_data = pd.read_csv("trajectories_filtered.csv")

# 创建 360 x 90 网格
lon_bins = np.arange(-180, 181, 2.5)  # 经度范围 -180° 到 180°（包含 180°）
lat_bins = np.arange(0, 91, 2.5)  # 纬度范围 0° 到 90°（北半球）

# 提取年份和月份
poi_data['datetime'] = poi_data['date'].apply(lambda x: datetime.strptime(str(x), "%Y%m%d%H"))
poi_data['year'] = poi_data['datetime'].apply(lambda x: x.year)
poi_data['month'] = poi_data['datetime'].apply(lambda x: x.month)

# 只保留冬季（12、1、2月）数据
# 注意：1月和2月属于自然年，而12月属于当前年，不需要调整
poi_data = poi_data[poi_data['month'].isin([12, 1, 2])].copy()

# 1月和2月的年份减1，12月保持原年份
poi_data.loc[poi_data['month'].isin([1, 2]), 'year'] -= 1

# 删除1959年的冬季数据（因为缺少1959年12月的数据）
poi_data = poi_data[~((poi_data['year'] == 1959) & (poi_data['month'].isin([12, 1, 2])))]

# 删除2024年的冬季数据（因为缺少2025年1月和2月的数据）
poi_data = poi_data[~((poi_data['year'] == 2024) & (poi_data['month'].isin([12, 1, 2])))]

# 调整 POI 经度为标准的 -180 到 180 范围
poi_data['center_lon'] = np.where(poi_data['center_lon'] > 180, poi_data['center_lon'] - 360, poi_data['center_lon'])

# 创建字典以便存储每个年份下 POI 数量
years = sorted(poi_data['year'].unique())  # 获取所有年份

# 初始化 NetCDF 文件
dataset = nc.Dataset('traj_winter.nc', 'w', format='NETCDF4')

# 创建经度、纬度、年份维度
dataset.createDimension('lon', len(lon_bins) - 1)
dataset.createDimension('lat', len(lat_bins) - 1)
dataset.createDimension('year', len(years))

# 创建经纬度和年份的变量
lon_var = dataset.createVariable('lon', np.float32, ('lon',))
lat_var = dataset.createVariable('lat', np.float32, ('lat',))
year_var = dataset.createVariable('year', np.int32, ('year',))

# 给经纬度和年份赋值
lon_var[:] = lon_bins[:-1] + 0.5  # 取中间点
lat_var[:] = lat_bins[:-1] + 0.5  # 取中间点
year_var[:] = np.array(years)

# 创建 POI 数量变量
poi_count = dataset.createVariable('traj', np.int32, ('year', 'lat', 'lon'))

# 初始化 POI 计数器
poi_count_data = np.zeros((len(years), len(lat_bins) - 1, len(lon_bins) - 1), dtype=int)

# 处理每一条 POI 数据
for _, row in poi_data.iterrows():
    lat_idx = np.digitize(row['center_lat'], lat_bins) - 1
    lon_idx = np.digitize(row['center_lon'], lon_bins) - 1
    year_idx = years.index(row['year'])

    # 防止索引超出范围，确保纬度和经度都在有效范围内
    lat_idx = min(max(lat_idx, 0), len(lat_bins) - 2)
    lon_idx = min(max(lon_idx, 0), len(lon_bins) - 2)

    # 增加对应栅格中的 POI 计数
    poi_count_data[year_idx, lat_idx, lon_idx] += 1

# 将 POI 数量数据写入 NetCDF 文件
poi_count[:] = poi_count_data

# 关闭文件
dataset.close()

print("NetCDF 文件已生成：traj_winter.nc")
