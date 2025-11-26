import xarray as xr

ssp_values = [126, 245, 370, 585]
index_values = ['prcptot', 'r95p', 'r99p', 'r95ptot', 'r99ptot', 'sdii', 'rx1day', 'rx5day', 'r10mm', 'r20mm', 'cwd']
for ssp in ssp_values:
    for index in index_values:
        datapath = f'E:/GEO/{ssp}{index}_mktest.nc'
        data = xr.open_dataset(datapath)
        data = data.rename({'level_1': 'lat', 'level_0': 'lon'})
        outputpath = f'E:/GEO/result/{ssp}{index}_mktest.nc'
        data.to_netcdf(outputpath)