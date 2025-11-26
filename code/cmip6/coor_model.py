import xarray as xr

datapath = f'E:/GEO/test/CHM_PRE_0.5dg_19612022.nc'
resultpath = f'E:/GEO/test/coor_model.nc'

# Open NetCDF file
data = xr.open_dataset(datapath)

# Resample data
resampled_data = data.coarsen(latitude=2, longitude=2, boundary='exact').mean()

# Save resampled data
resampled_data.to_netcdf(resultpath)

# Close NetCDF file
data.close()