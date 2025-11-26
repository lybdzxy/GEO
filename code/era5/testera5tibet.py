import xarray as xr

data = xr.open_dataset('E:/GEO/test/cut1951.nc')

days_in_month = data.time.dt.days_in_month
data = data*1000*days_in_month
annual_precip_data = data['tp'].resample(time='AS').sum(dim='time')
annual_precip_data.to_netcdf('E:/GEO/test/test.nc')