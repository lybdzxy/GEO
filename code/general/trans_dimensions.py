import xarray as xr
import tqdm

names = ['EC-Earth3', 'MPI-ESM1-2-HR', 'EC-Earth3-Veg',
         'CMCC-ESM2', 'GFDL-ESM4', 'INM-CM4-8',
         'BCC-CSM2-MR', 'MRI-ESM2-0', 'CMCC-CM2-SR5',
         'TaiESM1', 'NorESM2-MM', 'INM-CM5-0']
indices = ['prcptot', 'r95p', 'r99p', 'r95ptot', 'r99ptot', 'sdii', 'rx1day', 'r10mm', 'r20mm']

for name in tqdm.tqdm(names):
    for year in range(1988, 2015):
        for index in indices:
            data_path = f'E:/GEO/etccdi/qpm/{index}_{name}_historical_{year}.nc'
            data = xr.open_dataset(data_path)

            if 'time' in data.dims:
                data_transposed = data.transpose('time', 'lat', 'lon')
            else:
                data_transposed = data.transpose('lat', 'lon')

            data_transposed.to_netcdf(f'E:/GEO/etccdi/qpm/tr/{index}_{name}_historical_{year}.nc')
