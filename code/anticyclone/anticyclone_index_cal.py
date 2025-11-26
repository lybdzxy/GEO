import pandas as pd
import numpy as np
from global_land_mask import globe

# 读取轨迹数据
df = pd.read_csv('trajectories_filtered.csv')

# 将日期列转换为日期时间格式
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d%H')

# 按轨迹 ID 分组
grouped = df.groupby('trajectory_id')

# 初始化结果列表
results = []

# 对每个轨迹进行计算
for traj_id, group in grouped:
    # 提取数据
    lon = group['center_lon'].values  # 经度（0° 到 360°）
    lat = group['center_lat'].values  # 纬度
    t = group['date'].values          # 时间
    w = group['center_pressure'].values  # 权重（气压）

    # --- 计算加权平均经度 X_bar（圆形统计） ---
    lon_rad = np.deg2rad(lon)  # 转换为弧度
    sin_lon = np.sum(w * np.sin(lon_rad))
    cos_lon = np.sum(w * np.cos(lon_rad))
    X_bar = np.rad2deg(np.arctan2(sin_lon, cos_lon))  # 加权平均经度（弧度转度）
    if X_bar < 0:
        X_bar += 360  # 确保在 0° 到 360° 范围内

    # --- 计算加权平均纬度 Y_bar（圆形统计） ---
    lat_rad = np.deg2rad(lat)  # 转换为弧度
    sin_lat = np.sum(w * np.sin(lat_rad))
    cos_lat = np.sum(w * np.cos(lat_rad))
    Y_bar = np.rad2deg(np.arctan2(sin_lat, cos_lat))  # 加权平均纬度（度）

    # --- 计算时间相关指数 ---
    year = pd.Timestamp(t[0]).year
    year_start = np.datetime64(pd.Timestamp(year=year, month=1, day=1, hour=0))
    t_relative = (t - year_start).astype('timedelta64[h]').astype(float)  # 相对小时数
    T_bar = np.sum(w * t_relative) / np.sum(w)  # 加权平均时间
    T = (t[-1] - t[0]).astype('timedelta64[h]').astype(float)  # 时间跨度

    # --- 计算经度方差 Var_x（循环方差，无单位） ---
    R_lon = np.sqrt(sin_lon**2 + cos_lon**2) / np.sum(w)  # 经度平均向量长度
    Var_x = 1 - R_lon  # 无单位，范围 0 到 1

    # --- 计算纬度方差 Var_y（循环方差，无单位） ---
    R_lat = np.sqrt(sin_lat**2 + cos_lat**2) / np.sum(w)  # 纬度平均向量长度
    Var_y = 1 - R_lat  # 无单位，范围 0 到 1

    # --- 计算经纬度协方差 Var_xy（改进版，无单位，0 到 1） ---
    lon_diff = np.deg2rad(lon - X_bar)  # 经度偏差（弧度）
    lat_diff = np.deg2rad(lat - Y_bar)  # 纬度偏差（弧度）
    # 计算循环协方差的向量长度 R_xy
    sin_lon_diff = np.sin(lon_diff)
    cos_lon_diff = np.cos(lon_diff)
    sin_lat_diff = np.sin(lat_diff)
    cos_lat_diff = np.cos(lat_diff)
    # 加权平均的 sin 和 cos 乘积
    sin_sin = np.sum(w * sin_lon_diff * sin_lat_diff) / np.sum(w)
    cos_cos = np.sum(w * cos_lon_diff * cos_lat_diff) / np.sum(w)
    R_xy = np.sqrt(sin_sin**2 + cos_cos**2)  # 协方差的向量长度
    Var_xy = 1 - R_xy  # 无单位协方差，范围 0 到 1

    # --- 提取起始和结束经纬度 ---
    start_lon = group.iloc[0]['center_lon']
    start_lat = group.iloc[0]['center_lat']
    end_lon = group.iloc[-1]['center_lon']
    end_lat = group.iloc[-1]['center_lat']

    # --- 海陆判断 ---
    X_bar_for_ocean = X_bar if X_bar <= 180 else X_bar - 360  # 转换为 -180° 到 180°
    is_over_ocean = 1 if globe.is_ocean(Y_bar, X_bar_for_ocean) else 0

    # 调试信息：输出方差值以验证
    print(f"轨迹 {traj_id}: Var_x = {Var_x:.4f}, Var_y = {Var_y:.4f}, Var_xy = {Var_xy:.4f}")

    # 存储结果
    results.append({
        'trajectory_id': traj_id,
        'X_bar': X_bar,         # 加权平均经度（0° 到 360°）
        'Y_bar': Y_bar,         # 加权平均纬度（度）
        'Var_x': Var_x,         # 经度循环方差（无单位，0-1）
        'Var_y': Var_y,         # 纬度循环方差（无单位，0-1）
        'Var_xy': Var_xy,       # 经纬度协方差（无单位，0-1）
        'T_bar': T_bar,         # 加权平均时间（小时）
        'T': T,                 # 时间跨度（小时）
        'start_lon': start_lon,
        'start_lat': start_lat,
        'end_lon': end_lon,
        'end_lat': end_lat,
        'is_over_ocean': is_over_ocean
    })

# 转换为 DataFrame 并保存
results_df = pd.DataFrame(results)
results_df.to_csv('trajectory_statistics_tot.csv', index=False)

print("结果已成功保存到 'trajectory_statistics.csv' 文件中")