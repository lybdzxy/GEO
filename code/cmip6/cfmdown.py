import xarray as xr
import calendar
import numpy as np
import pandas as pd

for year in range(1950, 2015, 1):
    cmyearstart = ((year-1950)//5)*5+1950
    cmyearend = cmyearstart+4
    downyear=year+65
    era5_pattern = f'E:/GEO/ERA5/cor/cor{year}.nc'
    cmip6_pattern = f'E:/GEO/CMIP6/ssp/126/add/addredailymean{cmyearstart}_{cmyearend}.nc'
    obs = xr.open_dataset(era5_pattern)
    add = xr.open_dataset(cmip6_pattern)
    for months in range(1, 13):
        num_days_in_month = calendar.monthrange(year, months)[1]
        for days in range(1, num_days_in_month + 1):
            # Extract the data for the current month from obs and addre datasets
            obs_month_data = obs['tp'].sel(time=f'{year:04}-{months:02}-{days:02}')
            add_daily = add.sel(time=f'1951-{months:02}')

            # Add the data from obs to addre for the current month
            combined_data = (add_daily * 24 * 60 * 60 + obs_month_data * 1000).astype(np.float32)
            combined_data['time'] = pd.to_datetime([f'{downyear}-{months:02}-{days:02}'])
            combined_data = combined_data.drop_vars(['lon', 'lat'])

            # Save the result to a NetCDF file with a unique name for each month
            output_path = f'E:/GEO/downed/126/{downyear}_{months}_{days}.nc'
            combined_data.to_netcdf(output_path)