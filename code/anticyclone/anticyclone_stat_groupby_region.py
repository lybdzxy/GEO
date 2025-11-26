import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# 1. 读取 CSV 文件（输入文件使用制表符分隔）
csv_file = 'trajectory_statistics_tot.csv'
df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip()  # 清除列名中的前后空格
# 将经度从 0–360° 转换为 -180–180°
df['X_bar'] = df['X_bar'].apply(lambda x: x - 360 if x > 180 else x)

# 2. 加载 Shapefile 边界数据（假设文件名为 'global_boundaries.shp'）
shapefile_path = 'E:\GEO\geodata\World_Geographic_Regions\World_Geographic_Regionst.shp'  # 替换为您的 Shapefile 文件路径
gdf_boundaries = gpd.read_file(shapefile_path)
print(gdf_boundaries.columns)

# 3. 创建 GeoDataFrame，包含 CSV 数据的经纬度信息
geometry = [Point(lon, lat) for lon, lat in zip(df['X_bar'], df['Y_bar'])]
gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# 4. 确保 Shapefile 边界数据和点数据使用相同的坐标参考系统（CRS）
gdf_boundaries = gdf_boundaries.to_crs(gdf_points.crs)

# 5. 执行空间连接，确定每个点所属的区域
# 使用空间连接将点与边界匹配，'how="left"' 保留所有点信息，'op="within"' 过滤位于边界内的点
gdf_points = gpd.sjoin(gdf_points, gdf_boundaries, how='left', predicate='within')

# 6. 查看每个点所属的区域
print(gdf_points[['X_bar', 'Y_bar', 'Region']])  # 假设边界数据包含 'region_name' 列

# 7. 如果需要按区域分组，可以使用 Pandas 的 groupby 方法
df_grouped = gdf_points.groupby('Region')

# 8. 遍历所有区域并保存文件
for region, group in df_grouped:
    output_file_region = f'{region}_trajectories_stat.csv'
    group.to_csv(output_file_region, index=False)
    print(f"{region} 区域的数据已保存：", output_file_region)
