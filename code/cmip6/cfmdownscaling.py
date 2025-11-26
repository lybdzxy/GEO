import xarray as xr
import pandas as pd
import numpy as np
import calendar

his = xr.open_dataset(r'E:/testidm/pr_day_CESM2-WACCM_historical_r1i1p1f1_gn_19500101-19591231.nc')
mod = xr.open_dataset(r'E:/testidm/pr_day_CESM2-WACCM_ssp126_r1i1p1f1_gn_20150101-20241231.nc')
obs = xr.open_dataset(r'E:/I_LOVE_GEO/ERA5/1951.nc')

his_monthly_avr = his.resample(time='1M').mean()
mod_monthly_avr = mod.resample(time='1M').mean()

his_magm = his_monthly_avr.groupby('time.month')
mod_magm = mod_monthly_avr.groupby('time.month')

his_f10 = [his_magm[i].mean(dim='time') for i in range(1, 13)]
his_f10_array = xr.concat(his_f10, dim='month')
his_f10_array['month'] = np.arange(1, 13)
his_f10_array = his_f10_array.rename(month='time')
his_f10_array['time'] = pd.to_datetime(np.arange('1951-01', '1952-01', dtype='datetime64[M]'))

mod_f10 = [mod_magm[i].mean(dim='time') for i in range(1, 13)]
mod_f10_array = xr.concat(mod_f10, dim='month')
mod_f10_array['month'] = np.arange(1, 13)
mod_f10_array = mod_f10_array.rename(month='time')
mod_f10_array['time'] = pd.to_datetime(np.arange('1951-01', '1952-01', dtype='datetime64[M]'))

add = mod_f10_array['pr'] - his_f10_array['pr']
add.to_netcdf('add.nc')

# 选择obs数据的经度和纬度坐标
target_lon = obs['longitude']
target_lat = obs['latitude']

add_resampled = add.interp(lon=target_lon, lat=target_lat, method='nearest')
add_resampled['time'] = pd.to_datetime(np.arange('1951-01-01', '1952-01-01', dtype='datetime64[M]'))
add_resampled.to_netcdf('addre.nc')
# 获取每个月的天数
days_in_month = add_resampled.time.dt.days_in_month
# 将月降水数据差异平均分配到每一天
daily_avg_precipitation = add_resampled / days_in_month
daily_avg_precipitation.to_netcdf('daily_avg_precipitation.nc')

for months in range(1,13):
    num_days_in_month = calendar.monthrange(1951, months)[1]
    for days in range(1, num_days_in_month + 1):
         # Extract the data for the current month from obs and addre datasets
         obs_month_data = obs['tp'].sel(time=f'1951-{months:02}-{days:02}')
         daily_avg_precipitation_data = add_resampled.sel(time=f'1951-{months:02}')

         # Add the data from obs to addre for the current month
         combined_data = daily_avg_precipitation_data*24*60*60 + obs_month_data*1000

         # Save the result to a NetCDF file with a unique name for each month
         combined_data.to_netcdf(f'combined_{months}_{days}.nc')
'''
# Iterate through the months
for month in range(1, 13):
    # Extract the data for the current month from obs and addre datasets
    obs_month_data = obs['tp'].sel(time=f'1951-{month:02}')
    addre_month_data = add_resampled.sel(time=f'1951-{month:02}')

    # Add the data from obs to addre for the current month
    combined_data = addre_month_data + obs_month_data

    # Save the result to a NetCDF file with a unique name for each month
    combined_data.to_netcdf(f'combined_month_{month}.nc')
'''