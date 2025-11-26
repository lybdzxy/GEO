#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xarray as xr
import pandas as pd
from cftime import date2num
import re
import numpy as np
from netCDF4 import Dataset

# ==================== 配置 ====================
data_path = 'example_data/2021114567bj.nc'
time_var = 'valid_time'
output_path = 'example_data/2021114567bj_sh.nc'
# ================================================

# 1. 读取 + 解码
ds = xr.open_dataset(data_path, decode_times=False)
print(f"检测到时间变量: {time_var}")

orig_units = ds[time_var].attrs.get('units', '')
orig_calendar = ds[time_var].attrs.get('calendar', 'standard')
print(f"原始 units: {orig_units}")

decoded = xr.decode_cf(ds[[time_var]])
utc_dt = pd.to_datetime(decoded[time_var].values, utc=True)
utc8_dt = utc_dt.tz_convert('Asia/Shanghai').tz_localize(None)

# 2. 构造新 units
m = re.match(r'(seconds|hours|days) since ([\d\- :]+)', orig_units, re.I)
if m:
    unit_type = m.group(1).lower()
    ref_str = m.group(2)
    ref_dt = pd.to_datetime(ref_str)
    new_ref_dt = ref_dt + pd.Timedelta(hours=8)
    new_units = f'{unit_type} since {new_ref_dt.strftime("%Y-%m-%d %H:%M:%S")} +0800'
else:
    new_units = 'seconds since 1970-01-01 08:00:00 +0800'

print(f"新的 units: {new_units}")

# 3. 转为数值
new_time_num = date2num(utc8_dt.to_pydatetime(), units=new_units, calendar=orig_calendar)

# 4. 替换坐标
ds = ds.assign_coords({time_var: (time_var, new_time_num)})

# 5. 强制写 attrs
ds[time_var].attrs['units'] = new_units
ds[time_var].attrs['calendar'] = orig_calendar
ds[time_var].attrs['timezone'] = 'UTC+8'
ds[time_var].attrs['long_name'] = 'time (UTC+8)'

# 6. 先用 xarray 写（会丢 attrs）
ds.to_netcdf(output_path, mode='w')

# 7. 用 netCDF4 手动写回 attrs（关键！）
with Dataset(output_path, mode='r+') as nc:
    var = nc.variables[time_var]
    var.units = new_units
    var.calendar = orig_calendar
    var.timezone = 'UTC+8'
    var.long_name = 'time (UTC+8)'

print(f"已保存 → {output_path}")

# 8. 验证
ds_new = xr.open_dataset(output_path)
print("前5个时间:")
print(ds_new[time_var][:5].values)
print("units:", ds_new[time_var].attrs.get('units'))
print("calendar:", ds_new[time_var].attrs.get('calendar'))
ds_new.close()
ds.close()