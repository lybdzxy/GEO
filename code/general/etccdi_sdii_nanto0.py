import xarray as xr

names = ['EC-Earth3', 'MPI-ESM1-2-HR', 'EC-Earth3-Veg',
         'CMCC-ESM2', 'GFDL-ESM4', 'INM-CM4-8',
         'BCC-CSM2-MR', 'MRI-ESM2-0', 'CMCC-CM2-SR5',
         'TaiESM1', 'NorESM2-MM', 'INM-CM5-0']
for mod in names:
    for year in range(1988,2015):

        data_path = f'E:/GEO/etccdi/sdii{year}.nc'
        ds = xr.open_dataset(data_path)
        ds = ds.fillna(0)
        output_path = f'E:/GEO/etccdi/sdii/sdii{year}.nc'
        ds.to_netcdf(output_path)