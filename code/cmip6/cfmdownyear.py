import xarray as xr
import calendar
import numpy as np
import pandas as pd

for year in range(1950, 2015):
    cmyearstart = ((year-1950)//5)*5+1950
    cmyearend = cmyearstart+4
    downyear = year + 65
    era5_pattern = f'E:/GEO/ERA5/cor/cor{year}.nc'
    cmip6_pattern = f'E:/GEO/CMIP6/ssp/370/add/addredailymean{cmyearstart}_{cmyearend}.nc'
    obs = xr.open_dataset(era5_pattern)
    add = xr.open_dataset(cmip6_pattern)

    # Create empty lists
    combined_dates = []
    combined_values = []

    for months in range(1, 13):
        num_days_in_month = calendar.monthrange(year, months)[1]

        for days in range(1, num_days_in_month + 1):
            # Check if the date is valid (not 2017-02-29)
            if not (months == 2 and days == 29 and not calendar.isleap(downyear)):
                # Extract the data for the current day from obs and addre datasets
                obs_month_data = obs['tp'].sel(time=f'{year:04}-{months:02}-{days:02}')
                add_daily = add.sel(time=f'1951-{months:02}')

                # Calculate combined data for the current day
                combined_data = (add_daily * 24 * 60 * 60 + obs_month_data * 1000).astype(np.int16)

                # Append date and values to the lists
                combined_dates.append(pd.to_datetime(f'{downyear}-{months:02}-{days:02}'))
                combined_values.append(combined_data)

    # Merge the list of data arrays into a single data array
    combined_data_month = xr.concat(combined_values, dim='time')
    combined_data_month['time'] = combined_dates  # Add the time coordinate

    # Save the result to a NetCDF file for the entire year
    output_path = f'E:/GEO/downed/126/{downyear}.nc'
    combined_data_month.to_netcdf(output_path)
