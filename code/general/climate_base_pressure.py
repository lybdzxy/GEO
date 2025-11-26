import xarray as xr
ds_sp = []
for year in range(1940, 2025):
    data_path = rf'F:\ERA5\month\sfc\fin\ERA5_mon_sfc_{year}_moda_avgua.nc'
    data = xr.open_dataset(data_path)
    sp = data['sp']
    ds_sp.append(sp)
ds_sp = xr.concat(ds_sp, dim='valid_time')
ds_sp_monthly = ds_sp.groupby('valid_time.month').mean()
print(ds_sp_monthly)
ds_sp_monthly.to_netcdf('sp_climate_base.nc')