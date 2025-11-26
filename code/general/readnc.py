import xarray as xr
data=xr.open_dataset(r'E:/GEO/test/mod.nc')
print(data)
data_transposed = data.transpose('time', 'lat', 'lon')
data_transposed.to_netcdf(r'E:/GEO/test/mod_tr.nc')