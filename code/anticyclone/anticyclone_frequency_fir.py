import xarray as xr
import numpy as np
from scipy.signal import firwin, filtfilt
import pandas as pd

# === 读取 NetCDF 文件 ===
ds = xr.open_dataset("modified_file.nc")

# 将 month 字符串转为时间（如果不是时间格式）
if not np.issubdtype(ds['year'].dtype, np.datetime64):
    ds['year'] = pd.to_datetime(ds['year'].astype(str), format="%Y")

# 提取数据变量（假设变量名为 'traj'）
data = ds['traj_path']  # (year, lat, lon)
n_time, n_lat, n_lon = data.shape

# === 剔除前后11年数据
start_idx = 11
end_idx = n_time - 11
data_cropped = data[start_idx:end_idx, :, :]

# === 更新 'year' 维度的时间范围 ===
cropped_year = ds['year'].values[start_idx:end_idx]

# === 设计 FIR 滤波器 ===
window_length = 11
cutoff = 1 / 11
fir_coef = firwin(window_length, cutoff, window="hamming")

# === 初始化滤波后的数据数组 ===
filtered_data = np.full_like(data_cropped.values, np.nan, dtype=np.float32)

# === 对每个格点进行 FIR 滤波 ===
for i in range(n_lat):
    for j in range(n_lon):
        ts = data_cropped[:, i, j].values

        # 跳过全为缺失的点
        if np.all(np.isnan(ts)):
            continue

        # 插值填补缺失值（线性）
        ts_filled = pd.Series(ts).interpolate(limit_direction='both').values

        # 应用双向 FIR 滤波
        ts_filtered = filtfilt(fir_coef, [1.0], ts_filled)

        # 存入结果
        filtered_data[:, i, j] = ts_filtered

# === 构建新 Dataset 保存结果 ===
filtered_ds = xr.Dataset(
    {
        "traj_path": (("year", "lat", "lon"), filtered_data)
    },
    coords={
        "year": cropped_year,  # 更新为裁剪后的年份
        "lat": ds['lat'],
        "lon": ds['lon']
    }
)

# === 保存新文件 ===
filtered_ds.to_netcdf("m_f.nc")
print("✅ FIR 低通滤波完成，结果已保存为 traj_winter_fir.nc")