import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime

# ---------- 读取数据 ----------
poi_data = pd.read_csv("trajectories_zs.csv")

# ---------- 网格设置 ----------
lon_bins = np.arange(-180, 181, 2)
lat_bins = np.arange(0, 91, 2)

# ---------- 数据预处理 ----------
# 调整经度范围
poi_data['pressure_lon'] = np.where(poi_data['pressure_lon'] > 180, poi_data['pressure_lon'] - 360, poi_data['pressure_lon'])

# 提取年份
# poi_data['year'] = poi_data['date'].apply(lambda x: datetime.strptime(str(x), "%Y%m%d%H").year)
years = sorted(poi_data['date'].astype(str).str[0:3].astype(int).unique())

# ---------- 初始化 NetCDF ----------
dataset = nc.Dataset('traj_zs_paths_2deg.nc', 'w', format='NETCDF4')
dataset.createDimension('lon', len(lon_bins) - 1)
dataset.createDimension('lat', len(lat_bins) - 1)
dataset.createDimension('year', len(years))

# 经纬度和年份变量
lon_var = dataset.createVariable('lon', np.float32, ('lon',))
lat_var = dataset.createVariable('lat', np.float32, ('lat',))
year_var = dataset.createVariable('year', np.int32, ('year',))

lon_var[:] = lon_bins[:-1] + 0.25  # 中心点
lat_var[:] = lat_bins[:-1] + 0.25
year_var[:] = np.array(years)

# 创建路径格点计数字段
traj_path_count = dataset.createVariable('traj_path', np.int32, ('year', 'lat', 'lon'))
path_count_data = np.zeros((len(years), len(lat_bins) - 1, len(lon_bins) - 1), dtype=int)

# ---------- 轨迹处理 ----------
for traj_id, group in poi_data.groupby('trajectory_id'):
    group = group.sort_values('date'[0:3])
    coords = group[['pressure_lat', 'pressure_lon']].values
    year = group['date'[0:3]].iloc[0]
    year_idx = years.index(year)

    visited_cells = set()  # 每条轨迹一个集合，防止重复计数

    for i in range(len(coords) - 1):
        lat1, lon1 = coords[i]
        lat2, lon2 = coords[i + 1]

        # ------- 处理经度跳跃 -------
        lon_diff = lon2 - lon1
        if abs(lon_diff) > 180:
            if lon_diff > 0:
                lon2 -= 360  # 比如 170 到 -175，应看成 170 到 185
            else:
                lon2 += 360

        # 插值密度
        steps = max(abs(lat2 - lat1), abs(lon2 - lon1)) / 0.2
        for t in np.linspace(0, 1, int(steps) + 1):
            lat = lat1 + t * (lat2 - lat1)
            lon = lon1 + t * (lon2 - lon1)

            # 插值后经度修正回 [-180, 180]
            if lon > 180:
                lon -= 360
            elif lon < -180:
                lon += 360

            lat_idx = np.digitize(lat, lat_bins) - 1
            lon_idx = np.digitize(lon, lon_bins) - 1

            if 0 <= lat_idx < len(lat_bins) - 1 and 0 <= lon_idx < len(lon_bins) - 1:
                cell_key = (lat_idx, lon_idx)
                if cell_key not in visited_cells:
                    path_count_data[year_idx, lat_idx, lon_idx] += 1
                    visited_cells.add(cell_key)

# ---------- 写入 NetCDF ----------
traj_path_count[:] = path_count_data
dataset.close()
