import xarray as xr
import numpy as np
import pandas as pd
import glob
import os

# === 参数设置 ===
data_dir = r"F:\ERA5\month\sfc\fin"  # 数据所在文件夹
varname = "msl"  # 变量名
lat_range = (30, 70)  # 区域纬度范围（例如 20N-50N）
lon_range = (60, 130)  # 区域经度范围（例如 100E-150E）

# === 收集文件 ===
file_list = sorted(glob.glob(os.path.join(data_dir, "ERA5_mon_sfc_*_moda_avg.nc")))

# === 合并数据 ===
ds = xr.open_mfdataset(file_list, combine="by_coords")
da = ds[varname]

# === 子区域提取 ===
da_region = da.sel(latitude=slice(lat_range[1], lat_range[0]),
                   longitude=slice(lon_range[0], lon_range[1]))

# === 加权平均 ===
weights = np.cos(np.deg2rad(da_region.latitude))
weights.name = "weights"
da_weighted = da_region.weighted(weights).mean(dim=("latitude", "longitude"))

# === 转为 DataFrame，方便按年取冬季 ===
df = da_weighted.to_dataframe().reset_index()
df["year"] = df["valid_time"].dt.year
df["month"] = df["valid_time"].dt.month

# === 构造冬季指数 ===
winter_means = []
years = []

for year in range(1940, 2024):
    # 冬季 = 当年12月 + 次年1-2月
    sel = df[((df["year"] == year) & (df["month"] == 12)) |
             ((df["year"] == year + 1) & (df["month"].isin([1, 2])))]
    if len(sel) == 3:  # 确保三个月都有
        winter_means.append(sel[varname].mean())
        years.append(year)

winter_index = pd.Series(winter_means, index=years, name="winter_mslp")

print(winter_index.head())
winter_index.to_csv("winter_mslp_weighted_avg.csv")