import numpy as np
import xarray as xr
from datetime import datetime

data = xr.open_dataset(r'E:\GEO\pyproject\example_data\ERA5_18z_lvl_194001.nc')
da = data['z']
date = 1940010100
time_str = str(date).zfill(10)
year, month, day, hour = int(time_str[0:4]), int(time_str[4:6]), int(time_str[6:8]), int(time_str[8:10])
target_dt = datetime(year, month, day, hour)
da = da.sel(valid_time=target_dt, method='nearest')
da = da.sel(pressure_level=1000, method='nearest')

center_lon = 5
# --- 经度范围 ±24° ---
extent = 15
center_lon_norm = center_lon % 360
lon_min = (center_lon_norm - extent) % 360
lon_max = (center_lon_norm + extent) % 360

lon = da['longitude'].values
lon = np.mod(lon, 360)  # 确保 0-360

if lon_min < lon_max:
    # 不跨日期线
    result = da.sel(longitude=slice(lon_min, lon_max))
else:
    # 跨日期线：先统一坐标到中心在 0-360 范围内
    da_shifted = da.assign_coords(longitude=(da.longitude % 360))
    # 选取两段
    mask1 = da_shifted.longitude >= lon_min
    mask2 = da_shifted.longitude <= lon_max
    selected = da_shifted.where(mask1 | mask2, drop=True)

    # 再把跨日期线那部分 +360 让坐标连续
    lon_new = selected.longitude.copy()
    lon_new = xr.where(lon_new <= lon_max, lon_new + 360, lon_new)
    selected = selected.assign_coords(longitude=lon_new)
    result = selected.sortby('longitude')
result.to_netcdf('test.nc')
print(result)