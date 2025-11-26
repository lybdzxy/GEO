import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob
import os

# === 参数设置 ===
data_dir = r"F:\ERA5\month\lvl\new"   # 数据目录
file_pattern = os.path.join(data_dir, "ERA5_mon_lvl_*.nc")
g = 9.80665  # 重力加速度
lon_min, lon_max = 60, 130
lat_min, lat_max = 30, 70

# === 批量读取所有文件 ===
files = sorted(glob.glob(file_pattern))
if not files:
    raise FileNotFoundError("没有找到匹配文件，请检查路径和命名格式！")

print(f"找到 {len(files)} 个文件，示例: {files[0]}")

# 打开数据
ds = xr.open_mfdataset(files, combine="by_coords")

# 选择 500 hPa 位势高度 (m)
da = ds['z'].sel(pressure_level=500) / g

# === 挑选 DJF ===
months = da['valid_time'].dt.month
years = da['valid_time'].dt.year

# 构造 DJF 年份索引：12 月算作下一年的冬季
djf_year = xr.where(months == 12, years + 1, years)

# 把 djf_year 作为新坐标
da = da.assign_coords(djf_year=("valid_time", djf_year.data))

# 只保留 12/1/2 月份
da_djf = da.where((months == 12) | (months == 1) | (months == 2), drop=True)

# 只保留 1940–2024 的 DJF
da_djf = da_djf.where((da_djf.djf_year >= 1940) & (da_djf.djf_year <= 2024), drop=True)

# 按 DJF 年平均（每个冬季一张）
da_djf_group = da_djf.groupby("djf_year").mean("valid_time")

# 多年平均
da_djf_mean = da_djf_group.mean("djf_year")


# === 经纬度处理 ===
da_djf_mean = da_djf_mean.assign_coords(
    longitude=(((da_djf_mean.longitude + 180) % 360) - 180)
).sortby("longitude")

da_sub = da_djf_mean.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min))

# === 绘图 ===
plt.figure(figsize=(8, 6))
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=100))
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE.with_scale("50m"))
ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5)

# 填色
cs = ax.contourf(
    da_sub.longitude, da_sub.latitude, da_sub.squeeze(),
    levels=20, cmap="viridis", transform=ccrs.PlateCarree()
)
plt.colorbar(cs, orientation="vertical", pad=0.02, aspect=30, label="500 hPa 位势高度 (m)")

# 等值线
cs2 = ax.contour(
    da_sub.longitude, da_sub.latitude, da_sub.squeeze(),
    levels=np.arange(int(da_sub.min()), int(da_sub.max()), 60),
    colors="black", linewidths=0.6, transform=ccrs.PlateCarree()
)
ax.clabel(cs2, fmt="%d")

plt.title("1940–2024 DJF 平均 500 hPa 位势高度场（西伯利亚区域）")
plt.tight_layout()
plt.savefig("djf_500hpa_siberia.png", dpi=300)
plt.show()
