import pandas as pd

# 读取 trajectory_statistics_fin.csv 和 trajectories_with_season&region.csv 文件
trajectory_statistics = pd.read_csv('trajectory_statistics_fin.csv')
trajectories_with_season = pd.read_csv('trajectories_with_region_new.csv')
print(trajectories_with_season)
# 获取 trajectory_statistics 中的 trajectory_id 列
valid_trajectory_ids = trajectory_statistics['trajectory_id']

# 筛选出 trajectories_with_season&region.csv 中 trajectory_id 在 valid_trajectory_ids 中的行
filtered_trajectories = trajectories_with_season[trajectories_with_season['trajectory_id'].isin(valid_trajectory_ids)]

# 保存筛选后的数据到新的 CSV 文件
filtered_trajectories.to_csv('trajectories_fin_new.csv', index=False)

