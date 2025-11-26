import geopandas as gpd
import pandas as pd

# 读取 POI 数据（点数据）
poi = gpd.read_file("track_points.shp")
print(poi)

# 读取栅格数据（矢量化的网格）
grid = gpd.read_file("fishnet.shp")
print(grid.columns)
grid["grid_id"] = grid.index  # 使用索引作为唯一 ID

# 进行空间连接
joined = gpd.sjoin(poi, grid, how="left", predicate="within")

# 统计每个栅格中的 POI 数量
poi_count = joined.groupby("grid_id").size().reset_index(name="poi_count")

# 合并统计结果到原始网格
grid = grid.merge(poi_count, on="grid_id", how="left").fillna(0)

# 保存结果
grid.to_file("track_points_tot.shp")


