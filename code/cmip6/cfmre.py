import xarray as xr
import pandas as pd
import numpy as np
import calendar
obs = xr.open_dataset(r'E:/GEO/ERA5/1951.nc')
# 选择obs数据的经度和纬度坐标
target_lon = obs['longitude']
target_lat = obs['latitude']

add={}
add_resampled={}
addredaily={}
for d in range(6, 11, 1):
    for e in range(1950, 2011, 5):
        f = e + 5 - 1
        add_path = f'E:/GEO/CMIP6/add/add{d}_{e}_{f}.nc'
        add[d, e] = xr.open_dataset(add_path)
        add_resampled[d, e] = add[d, e].interp(lon=target_lon, lat=target_lat, method='nearest')
        add_resampled[d, e]['time'] = pd.to_datetime(np.arange('1951-01-01', '1952-01-01', dtype='datetime64[M]'))
        # 获取每个月的天数
        days_in_month = add_resampled[d, e].time.dt.days_in_month
        # 将月降水数据差异平均分配到每一天
        addredaily[d, e] = add_resampled[d, e] / days_in_month
        output_path = f'E:/GEO/CMIP6/addredaily/addredaily{d}_{e}_{f}.nc'
        addredaily[d, e].to_netcdf(output_path)
