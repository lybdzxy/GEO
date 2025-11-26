import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import xarray as xr

# === 读取 NetCDF 文件 ===
ds = xr.open_dataset("modified_file.nc")

# 提取数据变量（假设变量名为 'traj_path'）
data = ds['traj_path']  # (year, lat, lon)
n_time, n_lat, n_lon = data.shape

# === 设计巴特沃斯高通滤波器 ===
cutoff = 1 / 11  # 截止频率
order = 4  # 滤波器阶数
b, a = butter(order, cutoff, btype='high', fs=1)  # 设计高通巴特沃斯滤波器

# === 初始化滤波后的数据数组 ===
filtered_data = np.full_like(data.values, np.nan, dtype=np.float64)

# === 对每个格点进行巴特沃斯高通滤波 ===
for i in range(n_lat):
    for j in range(n_lon):
        ts = data[:, i, j].values

        # 跳过全为缺失的点
        if np.all(np.isnan(ts)):
            continue

        # 插值填补缺失值（线性）
        ts_filled = pd.Series(ts).interpolate(limit_direction='both').values

        # 应用巴特沃斯高通滤波
        ts_filtered = filtfilt(b, a, ts_filled)

        # 将滤波后的数据写入：保留原有NaN
        ts_filtered_with_nan = np.copy(ts_filtered)
        ts_filtered_with_nan[np.isnan(ts)] = np.nan  # 恢复NaN

        # 防止小负值被误替代为0
        ts_filtered_with_nan[ts_filtered_with_nan == 0] = np.nan  # 将极小负数或接近零的数设置为 NaN

        # 存入结果
        filtered_data[:, i, j] = ts_filtered_with_nan

# === 去掉首尾各一年的数据 ===
filtered_data_cropped = filtered_data[1:-1, :, :]  # 去掉第一个和最后一个时间点的数据

# === 更新 'year' 维度的时间范围 ===
cropped_year = ds['year'].values[1:-1]  # 去掉首尾年份

# === 构建新 Dataset 保存结果 ===
filtered_ds = xr.Dataset(
    {
        "traj_path": (("year", "lat", "lon"), filtered_data_cropped)
    },
    coords={
        "year": cropped_year,  # 使用裁剪后的年份
        "lat": ds['lat'],
        "lon": ds['lon']
    }
)
print(filtered_ds['traj_path'])
# === 保存新文件 ===
filtered_ds.to_netcdf("m_b.nc")
print("✅ 巴特沃斯高通滤波完成，首尾数据已剔除，结果已保存为 filtered_file.nc")
