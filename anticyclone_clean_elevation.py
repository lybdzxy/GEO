import pandas as pd
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree

# 1. 读取 CSV 文件（输入文件使用制表符分隔）
csv_file = 'trajectories_with_pressure6024_tot_cleaned.csv'
df = pd.read_csv(csv_file)
# 清除列名中的前后空格
df.columns = df.columns.str.strip()
print("DataFrame 的列名：", df.columns.tolist())

# 2. 打开 ETOPO DEM 数据（NetCDF 格式）
dem_file = 'ETOPO_2022_v1_60s_N90W180_bed.nc'
ds = xr.open_dataset(dem_file)

# 假设 DEM 数据中经纬度变量名称为 'lon' 和 'lat'，海拔数据为 'z'
lons = ds['lon'].values  # 经度数组
lats = ds['lat'].values  # 纬度数组
elevation_data = ds['z'].values  # 海拔数据

# 3. 构造 DEM 网格的经纬度点，并建立 KDTree
lon_grid, lat_grid = np.meshgrid(lons, lats)  # 注意维度顺序为 (lat, lon)
grid_points = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
tree = cKDTree(grid_points)

# 4. 利用 KDTree 查找 CSV 中每个点的最近邻格点，获取海拔值
csv_points = np.column_stack((df['center_lat'].values, df['center_lon'].values))
dist, idx = tree.query(csv_points)
df['elevation'] = elevation_data.ravel()[idx]

# 5. 按 trajectory_id 分组，过滤掉任一数据点海拔超过 2500 米的整个轨迹
max_elev = df.groupby('trajectory_id')['elevation'].max()
bad_ids = max_elev[max_elev > 2500].index.tolist()
df_filtered = df[~df['trajectory_id'].isin(bad_ids)]

# 6. 输出 CSV 文件，默认逗号分隔（分列输出）
output_file = 'trajectories_filtered.csv'
df_filtered.to_csv(output_file, index=False)
print("筛选完成，结果保存在", output_file)
