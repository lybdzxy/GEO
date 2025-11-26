import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ======== 1. 数据读取 ========
# 支持多文件：files = sorted(glob("path/to/msl_monthly_*.nc"))
# 或单文件包含多年月：
ds = xr.open_mfdataset(
    r"F:\ERA5\month\sfc\fin\ERA5_mon_sfc_*_moda_avgua.nc",  # 改成你的路径/通配
    combine="by_coords"
)

# 统一变量名（常见是 'msl'；若是别名请改）
da = ds["msl"]  # (time, lat, lon), units: Pa
# 确保经纬度命名一致
if "longitude" in da.dims:
    da = da.rename({"longitude": "lon"})
if "latitude" in da.dims:
    da = da.rename({"latitude": "lat"})
if "valid_time" in da.dims:
    da = da.rename({"valid_time": "time"})

# 处理经度到 [-180, 180)
if (da.lon.max() > 180).item():
    lon_new = ((da.lon + 180) % 360) - 180
    da = da.assign_coords(lon=lon_new).sortby("lon")

# ======== 2. 选择季节与区域 ========
# 常用 DJF；如要 NDJFM, 把 months 改为 [11,12,1,2,3]
months = [12, 1, 2]   # DJF
da = da.sel(lat=slice(70, 30), lon=slice(60, 130))  # 北半球从高到低是 90->-90，故 lat 用 slice(高,低)

# 仅取所需月份
da = da.sel(time=np.isin(da["time.month"], months))

# 为了把 DJF 归到“年”，我们令 12 月归入下一年
time = pd.DatetimeIndex(da.time.values)
year_for_season = time.year + (time.month == 12)  # 12月归到下一年
da = da.assign_coords(season_year=("time", year_for_season))

# ======== 3A. 中心法：逐月找极大值 -> 对季节/多年平均 ========
# 找每个 time 的最大值位置（SH 中心）
def find_max_xy(da2d):
    arr = da2d.values
    idx_flat = arr.argmax()
    lat_idx, lon_idx = np.unravel_index(idx_flat, arr.shape)
    lat = da2d["lat"].values[lat_idx]
    lon = da2d["lon"].values[lon_idx]
    val = arr[lat_idx, lon_idx]
    return xr.Dataset({"lat": ([], lat), "lon": ([], lon), "msl": ([], val)})

centers = xr.concat([find_max_xy(da.sel(time=t)) for t in da.time], dim="time")
centers = centers.assign_coords(time=da.time)


# 按季节年聚合（例如对所有 DJF 的“中心点”求多年平均）
center_mean = centers.groupby("season_year").mean()
mean_center_over_years = center_mean.mean(dim="season_year")  # 多年平均中心
mean_center_lat = float(mean_center_over_years["lat"].values)
mean_center_lon = float(mean_center_over_years["lon"].values)

# ======== 3B. 等压线法（可选）：在多年 DJF 平均场上画阈值等压线 ========
clim_djf = da.groupby("season_year").mean("time").mean("season_year")  # 多年DJF平均场（Pa）
clim_hpa = clim_djf / 100.0
isobar = 1028.0  # 常见阈值，可改 1026–1030 做敏感性
# 简单“加权质心”（气压超阈值作为权重）
mask = (clim_hpa >= isobar)
if mask.any():
    LAT, LON = np.meshgrid(clim_hpa["lat"].values, clim_hpa["lon"].values, indexing="ij")
    w = np.clip((clim_hpa - isobar).where(mask, 0.0).values, 0, None)
    w_sum = w.sum()
    if w_sum > 0:
        cm_lat = float((w * LAT).sum() / w_sum)
        cm_lon = float((w * LON).sum() / w_sum)
    else:
        cm_lat, cm_lon = np.nan, np.nan
else:
    cm_lat, cm_lon = np.nan, np.nan

# ======== 4. 作图 ========
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(8.5, 6.5))
ax = plt.axes(projection=proj)
ax.set_extent([60, 140, 30, 70], crs=proj)
ax.coastlines(linewidth=0.8)
ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6)
ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=0.3)
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 等压线：多年 DJF 平均场
cs = ax.contour(
    clim_hpa["lon"], clim_hpa["lat"], clim_hpa,
    levels=np.arange(1000, 1052, 2), linewidths=0.6, transform=proj
)
ax.clabel(cs, fmt="%.0f", fontsize=7)

# 画阈值等压线（加粗）
ax.contour(
    clim_hpa["lon"], clim_hpa["lat"], clim_hpa,
    levels=[isobar], linewidths=1.5, transform=proj
)

# 标出“中心法”的多年平均中心
ax.plot(mean_center_lon, mean_center_lat, marker="o", markersize=6, transform=proj)
ax.text(mean_center_lon+1, mean_center_lat+0.8, "中心法: {:.1f}°N, {:.1f}°E".format(
    mean_center_lat, mean_center_lon),
    fontsize=9, transform=proj
)

# 标出“等压线法”的加权质心（若可用）
if np.isfinite(cm_lat):
    ax.plot(cm_lon, cm_lat, marker="^", markersize=6, transform=proj)
    ax.text(cm_lon+1, cm_lat+0.8, "等压线法: {:.1f}°N, {:.1f}°E".format(
        cm_lat, cm_lon),
        fontsize=9, transform=proj
    )

ax.set_title("ERA5 多年 DJF 海平面气压与西伯利亚高压平均位置（{} hPa 等压线）".format(int(isobar)))
plt.tight_layout()
plt.show()
