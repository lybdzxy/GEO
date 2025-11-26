import pandas as pd
id_data_path = 'anticyclone_tracks_siberia_majority.csv'
data_id = pd.read_csv(id_data_path)
data_path = 'trajectory_statistics_fin.csv'
data = pd.read_csv(data_path)

# === 获取路径数据中存在的 trajectory_id ===
valid_ids = data_id["trajectory_id"].unique()

# === 筛选参数数据，只保留存在的 id ===
df_params_filtered = data[data["trajectory_id"].isin(valid_ids)]

# === 保存结果 ===
df_params_filtered.to_csv("anticyclone_tracks_siberia_majority_statistics.csv", index=False)