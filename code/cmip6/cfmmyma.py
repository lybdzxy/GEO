import xarray as xr
import pandas as pd
import numpy as np
import calendar

obs = xr.open_dataset(r'E:/GEO/ERA5/1951.nc')

# Choose longitude and latitude coordinates from the 'obs' dataset
target_lon = obs['longitude']
target_lat = obs['latitude']

# Create dictionaries to store the datasets
his = {}
hisre = {}
s585sp = {}
s585spre = {}
his_monthly_avr = {}
mod_monthly_avr = {}
his_magm = {}
mod_magm = {}
his_f10 = {}
his_f10_array = {}
mod_f10 = {}
mod_f10_array = {}
add = {}
for d in range(1, 11, 1):
    for e in range(1950, 2011, 5):
        f = e + 5 - 1
        g = e+65
        m = g+5-1
        file_path_his = f'E:/GEO/CMIP6/his/split/his{d}_{e}_{f}.nc'
        file_path_s585sp = f'E:/GEO/CMIP6/ssp/585/split/s585sp{d}_{g}_{m}.nc'
        his[d, e] = xr.open_dataset(file_path_his)
        s585sp[d, g] =xr.open_dataset(file_path_s585sp)
        #xiugai
        his_monthly_avr[d, e] = his[d, e].resample(time='1M').mean()
        mod_monthly_avr[d, g] = s585sp[d, g].resample(time='1M').mean()

        his_magm[d, e] = his_monthly_avr[d, e].groupby('time.month')
        mod_magm[d, g] = mod_monthly_avr[d, g].groupby('time.month')

        his_f10[d, e] = [his_magm[d, e][a].mean(dim='time') for a in range(1, 13)]
        his_f10_array[d, e] = xr.concat(his_f10[d, e], dim='month')
        his_f10_array[d, e]['month'] = np.arange(1, 13)
        his_f10_array[d, e] = his_f10_array[d, e].rename(month='time')
        his_f10_array[d, e]['time'] = pd.to_datetime(np.arange('1951-01', '1952-01', dtype='datetime64[M]'))

        mod_f10[d, g] = [mod_magm[d, g][a].mean(dim='time') for a in range(1, 13)]
        mod_f10_array[d, g] = xr.concat(mod_f10[d, g], dim='month')
        mod_f10_array[d, g]['month'] = np.arange(1, 13)
        mod_f10_array[d, g] = mod_f10_array[d, g].rename(month='time')
        mod_f10_array[d, g]['time'] = pd.to_datetime(np.arange('1951-01', '1952-01', dtype='datetime64[M]'))

        add[d, e] = mod_f10_array[d, g]['pr'] - his_f10_array[d, e]['pr']
        output_path = f'E:/GEO/CMIP6/add/add{d}_{e}_{f}.nc'
        add[d, e].to_netcdf(output_path)