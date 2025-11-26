import xarray as xr
import numpy as np

names = ['precipitation_3_year', 'precipitation_10_year', 'precipitation_20_year', 'precipitation_50_year',
         'precipitation_100_year']
indices = ['r95p', 'r99p', 'rx1day']
ssps = ['126', '245', '370', '585']
lon = 87.25
lat = 39.25
for ssp in ssps:
    for index in indices:
        for name in names:
            data_path = f'E:/GEO/result/qpm/gev/{ssp}{index}_{name}.nc'
            data = xr.open_dataset(data_path)

            output_path = f'E:/GEO/result/qpm/gev/{ssp}{index}_{name}.nc'
            data = data.where(~((data.lon == lon) & (data.lat == lat)),
                                  other=float('nan'))
            data.to_netcdf(output_path)
