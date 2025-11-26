import xarray as xr
import xclim.indices as indices
import numpy as np
import dask
from dask import delayed
import dask.threaded
import dask.multiprocessing

def calculate_indices(year):
    input_path = f'E:/GEO/chm_pre/CHM_PRE{year}.nc'
    ds = xr.open_dataset(input_path)
    pr = ds['pre']
    pr.attrs['units'] = 'mm/day'

    prcp = indices.prcptot(pr)
    prcp.to_netcdf(f'E:/GEO/etccdi/chm_pre/prcptot/prcptot{year}.nc')

    p95 = pr.quantile(0.95, dim="time", keep_attrs=True)
    r95p = indices._anuclim.prcptot(pr, p95)
    r95p.to_netcdf(f'E:/GEO/etccdi/chm_pre/r95p/r95p{year}.nc')

    p99 = pr.quantile(0.99, dim="time", keep_attrs=True)
    r99p = indices._anuclim.prcptot(pr, p99)
    r99p.to_netcdf(f'E:/GEO/etccdi/chm_pre/r99p/r99p{year}.nc')

    r95ptot = r95p / prcp
    r95ptot.to_netcdf(f'E:/GEO/etccdi/chm_pre/r95ptot/r95ptot{year}.nc')
    r99ptot = r99p / prcp
    r99ptot.to_netcdf(f'E:/GEO/etccdi/chm_pre/r99ptot/r99ptot{year}.nc')

    sdii = indices.daily_pr_intensity(pr)
    sdii.to_netcdf(f'E:/GEO/etccdi/chm_pre/sdii/sdii{year}.nc')

    rx1day = indices.max_n_day_precipitation_amount(pr, window=1)
    rx1day.to_netcdf(f'E:/GEO/etccdi/chm_pre/rx1day/rx1day{year}.nc')

    rx5day = indices.max_n_day_precipitation_amount(pr, window=5)
    rx5day.to_netcdf(f'E:/GEO/etccdi/chm_pre/rx5day/rx5day{year}.nc')

    r10mm = ((pr > 10).groupby('time.dayofyear').sum(dim='time') > 0).sum(dim='dayofyear')
    r10mm.to_netcdf(f'E:/GEO/etccdi/chm_pre/r10mm/r10mm{year}.nc')

    r20mm = ((pr > 20).groupby('time.dayofyear').sum(dim='time') > 0).sum(dim='dayofyear')
    r20mm.to_netcdf(f'E:/GEO/etccdi/chm_pre/r20mm/r20mm{year}.nc')

    cwd = indices.wet_spell_max_length(pr)
    cwd.to_netcdf(f'E:/GEO/etccdi/chm_pre/cwd/cwd{year}.nc')

# Create a list of delayed objects for each year
delayed_tasks = [delayed(calculate_indices)(year) for year in range(1961, 2023)]

# Execute the delayed tasks in parallel using Dask
results = dask.compute(*delayed_tasks, scheduler='threads')  # or 'processes' for multiprocessing

# The results variable will contain the computed values
