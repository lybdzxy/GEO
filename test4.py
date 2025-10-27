import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

# 1. 读取 CSV 文件（输入文件使用制表符分隔）
csv_file = 'trajectories_filtered_season.csv'
df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip()  # 清除列名中的前后空格

# 调试步骤 1：验证点坐标有效性
print("调试步骤 1：检查点坐标有效性")
print(f"经度范围：{df['center_lon'].min()} 至 {df['center_lon'].max()}")
print(f"纬度范围：{df['center_lat'].min()} 至 {df['center_lat'].max()}")
if not df['center_lon'].between(-180, 180).all():
    print("警告：存在无效经度值（应在 -180 至 180 之间）")
if not df['center_lat'].between(-90, 90).all():
    print("警告：存在无效纬度值（应在 -90 至 90 之间）")
if df[['center_lon', 'center_lat']].isna().any().any():
    print("警告：点坐标中存在 NaN 值")
# 调整经度（如果需要）
df['center_lon'] = df['center_lon'].apply(lambda x: x - 360 if x > 180 else x)

# 2. 加载 Shapefile 边界数据
shapefile_path = 'E:/GEO/geodata/World_Geographic_Regions/World_Geographic_Regionst.shp'  # 修正可能的拼写错误
gdf_boundaries = gpd.read_file(shapefile_path)

# 调试步骤 2：检查 Shapefile 的 CRS 和列
print("\n调试步骤 2：检查 Shapefile 的 CRS 和列")
print(f"Shapefile CRS: {gdf_boundaries.crs}")
print(f"Shapefile 列: {gdf_boundaries.columns.tolist()}")
print(f"前几个 Region 值:\n{gdf_boundaries['Region'].head()}")

# 调试步骤 3：检查无效几何
print("\n调试步骤 3：检查 Shapefile 中无效几何")
invalid_geometries = gdf_boundaries[~gdf_boundaries.geometry.is_valid]
if not invalid_geometries.empty:
    print(f"警告：发现 {len(invalid_geometries)} 个无效几何")
    print(invalid_geometries[['Region', 'geometry']])
    # 尝试修复无效几何
'''    gdf_boundaries['geometry'] = gdf_boundaries.geometry.buffer(0)
    print(f"已尝试修复无效几何，重新检查: {gdf_boundaries.is_valid.all()}")'''

# 3. 创建点数据的 GeoDataFrame
geometry = [Point(lon, lat) for lon, lat in zip(df['center_lon'], df['center_lat'])]
gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# 4. 确保 CRS 一致
print("\n调试步骤 4：确保 CRS 一致")
if gdf_boundaries.crs != gdf_points.crs:
    print(f"Shapefile CRS ({gdf_boundaries.crs}) 与点数据 CRS ({gdf_points.crs}) 不一致，正在转换")
    gdf_boundaries = gdf_boundaries.to_crs(gdf_points.crs)
    print(f"转换后 Shapefile CRS: {gdf_boundaries.crs}")

# 5. 执行空间连接
gdf_points = gpd.sjoin(gdf_points, gdf_boundaries, how='left', predicate='within')

# 调试步骤 5：检查重复分配
print("\n调试步骤 5：检查空间连接后的重复分配")
duplicates = gdf_points.index.duplicated().sum()
if duplicates > 0:
    print(f"警告：发现 {duplicates} 个点被分配到多个区域")
    # 处理重复：保留第一个匹配
    gdf_points = gdf_points[~gdf_points.index.duplicated(keep='first')]
    print("已保留第一个匹配区域")

# 6. 处理区域名称
print("\n调试步骤 6：检查区域列格式")
if 'Region' in gdf_points.columns:
    gdf_points['region_name'] = gdf_points['Region'].str.split(',').str[0]
    print(f"前几个 region_name 值:\n{gdf_points['region_name'].head()}")
else:
    print("错误：空间连接后未找到 'Region' 列，请检查 Shapefile 列名")
    print(f"可用列: {gdf_points.columns.tolist()}")
    exit()

# 调试步骤 7：测试已知点
print("\n调试步骤 7：测试已知点")
# 示例：测试纽约 (经度: -74.0060, 纬度: 40.7128) 和巴黎 (经度: 2.3522, 纬度: 48.8566)
test_points = pd.DataFrame({
    'center_lon': [-74.0060, 2.3522],
    'center_lat': [40.7128, 48.8566],
    'name': ['New York', 'Paris']
})
test_gdf = gpd.GeoDataFrame(
    test_points,
    geometry=[Point(lon, lat) for lon, lat in zip(test_points['center_lon'], test_points['center_lat'])],
    crs="EPSG:4326"
)
test_result = gpd.sjoin(test_gdf, gdf_boundaries, how='left', predicate='within')
test_result['region_name'] = test_result['Region'].str.split(',').str[0]
print("已知点测试结果:")
print(test_result[['name', 'region_name']])

# 7. 将结果合并回原始 DataFrame
df = df.reset_index(drop=True)
gdf_points = gdf_points.reset_index(drop=True)
df['region_name'] = gdf_points['region_name']

# 8. 输出结果
output_file = 'trajectories_with_region_new.csv'
df.to_csv(output_file, index=False)
print(f"\n结果已保存至 {output_file}")