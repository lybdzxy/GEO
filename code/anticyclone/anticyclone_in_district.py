import pandas as pd

# 读取数据
df = pd.read_csv("trajectories_fin_new.csv")

# 定义西伯利亚区域范围
lon_min, lon_max = 60, 130
lat_min, lat_max = 30, 70

# 判断每个点是否在西伯利亚范围内
df["in_siberia"] = (
    (df["center_lon"] >= lon_min) & (df["center_lon"] <= lon_max) &
    (df["center_lat"] >= lat_min) & (df["center_lat"] <= lat_max)
)

# 统计每条路径的点数和在西伯利亚的点数
path_stats = df.groupby("trajectory_id")["in_siberia"].agg(
    total_points="count",
    siberia_points="sum"
)

# 计算比例
path_stats["ratio"] = path_stats["siberia_points"] / path_stats["total_points"]

# 找出满足条件的路径 ID
valid_ids = path_stats.loc[path_stats["ratio"] > 0.5].index

# 筛选出这些路径的所有点
df_siberia_majority = df[df["trajectory_id"].isin(valid_ids)]

# 保存结果
df_siberia_majority.to_csv("anticyclone_tracks_siberia_majority.csv", index=False)