import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# 1. 读取 CSV 文件（输入文件使用制表符分隔）
csv_file = 'trajectories_filtered_season.csv'
df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip()  # 清除列名中的前后空格
df['center_lon'] = df['center_lon'].apply(lambda x: x - 360 if x > 180 else x)

# 2. 加载 Shapefile 边界数据（假设文件名为 'global_boundaries.shp'）
shapefile_path = 'E:/GEO/geodata/World_Geographic_Regions/World_Geographic_Regionst.shp'  # 替换为您的 Shapefile 文件路径
gdf_boundaries = gpd.read_file(shapefile_path)
print(gdf_boundaries.columns)  # 查看 Shapefile 中的列名

# 3. 创建 GeoDataFrame，包含 CSV 数据的经纬度信息
geometry = [Point(lon, lat) for lon, lat in zip(df['center_lon'], df['center_lat'])]
gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# 4. 确保 Shapefile 边界数据和点数据使用相同的坐标参考系统（CRS）
gdf_boundaries = gdf_boundaries.to_crs(gdf_points.crs)

# 5. 执行空间连接，确定每个点所属的区域
# 使用空间连接将点与边界匹配，'how="left"' 保留所有点信息，'predicate="within"' 过滤位于边界内的点
gdf_points = gpd.sjoin(gdf_points, gdf_boundaries, how='left', predicate='within')

# 6. 处理可能的重复区域名称
gdf_points['region_name'] = gdf_points['Region'].str.split(',').str[0]

# 7. 将 region_name 添加到原始 DataFrame
df = df.reset_index(drop=True)
gdf_points = gdf_points.reset_index(drop=True)
df['region_name'] = gdf_points['region_name']

# 8. 输出结果
output_file = 'trajectories_with_season&region_new.csv'
df.to_csv(output_file, index=False)
print(f"结果已保存至 {output_file}")