import pandas as pd
import math

# 定义 haversine 函数，计算两点之间的距离（单位：公里）
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # 地球半径（公里）
    return c * r

# 读取 CSV 数据，假设包含 date, center_lon, center_lat, center_pressure 等字段
df = pd.read_csv("centers_zs_cw636_ele.csv")

# 确保 date 字段为字符串类型
df['date'] = df['date'].astype(str)

# 筛选北纬20°以北的点
df_filtered = df[df['pressure_lat'] >= 20].copy()

# 按日期排序
df_filtered.sort_values(by='date', inplace=True)

# 按日期分组
groups = df_filtered.groupby('date')
time_groups = {}
for date, group in groups:
    time_groups[date] = group.reset_index(drop=True)

# 获取所有时刻
time_list = sorted(time_groups.keys())

# 存储最终轨迹
final_trajectories = []
prev_points = dict()
unique_id = 0

# 遍历各时刻，构建轨迹
for i, current_time in enumerate(time_list):
    current_group = time_groups[current_time]
    current_points = dict()
    matched_current = set()

    if i == 0:
        # 初始时刻，直接为每个点分配轨迹ID
        for idx, row in current_group.iterrows():
            prev_points[unique_id] = [row.to_dict()]
            unique_id += 1
    else:
        # 后续时刻，匹配前一时刻的点
        for pid, traj in prev_points.items():
            last_point = traj[-1]
            min_dist = float('inf')
            min_idx = None
            for idx, row in current_group.iterrows():
                if idx in matched_current:
                    continue
                dist = haversine(last_point['pressure_lon'], last_point['pressure_lat'],
                                 row['pressure_lon'], row['pressure_lat'])
                if dist < min_dist:
                    min_dist = dist
                    min_idx = idx
            if min_idx is not None and min_dist <= 1280:
                # 如果距离小于1280公里，延续轨迹
                new_traj = traj + [current_group.loc[min_idx].to_dict()]
                current_points[pid] = new_traj
                matched_current.add(min_idx)
            else:
                # 如果轨迹长度>=4，保存到最终轨迹
                if len(traj) >= 4:
                    final_trajectories.append(traj)
        # 处理未匹配的点，创建新轨迹
        for idx, row in current_group.iterrows():
            if idx not in matched_current:
                current_points[unique_id] = [row.to_dict()]
                unique_id += 1
        prev_points = current_points

# 处理最后一时刻的轨迹
for traj in prev_points.values():
    if len(traj) >= 4:
        final_trajectories.append(traj)

# 将轨迹数据写入 CSV 文件，包含中心气压
trajectory_data = []
for traj_id, traj in enumerate(final_trajectories):
    for point in traj:
        data = {
            'trajectory_id': traj_id,
            'date': point['date'],
            'pressure_lon': point['pressure_lon'],
            'pressure_lat': point['pressure_lat'],
            'pressure_value': point['pressure_value'],
            'stream_lon': point['stream_lon'],
            'stream_lat': point['stream_lat'],
            'stream_value': point['stream_value'],
            'system_type': point['system_type'],
            'ratio': point['ratio'],
            'elevation': point['elevation']
        }
        trajectory_data.append(data)

# 创建 DataFrame 并写入 CSV
df_trajectories = pd.DataFrame(trajectory_data)
df_trajectories.to_csv('trajectories_zs.csv', index=False)
print("轨迹数据（包含中心气压）已成功写入 'trajectories_zs.csv' 文件中")