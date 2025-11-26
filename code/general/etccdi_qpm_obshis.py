import xarray as xr
import xclim.indices as indices
import numpy as np
import concurrent.futures
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

names = ['EC-Earth3', 'MPI-ESM1-2-HR', 'EC-Earth3-Veg',
         'CMCC-ESM2', 'GFDL-ESM4', 'INM-CM4-8',
         'BCC-CSM2-MR', 'MRI-ESM2-0', 'CMCC-CM2-SR5',
         'TaiESM1', 'NorESM2-MM', 'INM-CM5-0']

def process_data(mod, year):
    input_path = f'E:/GEO/down/qpm/downed_{mod}_historical_{year}.nc'
    # 打开包含 ERA5 数据的 NetCDF 文件
    ds = xr.open_dataset(input_path)
    # 选择特定的经度和纬度范围
    pr = ds['__xarray_dataarray_variable__'].assign_coords(time=ds['time'])

    # 提取降水数据，假设数据变量名为 'pr'（总降水量，单位为 mm/day）
    pr.attrs['units'] = 'mm/day'

    #etccdi计算
    # 计算 PRCPTOT（年降水总量）
    prcp = indices.prcptot(pr)
    prcp.to_netcdf(f'E:/GEO/etccdi/qpm/prcptot_{mod}_historical_{year}.nc')
    # 计算 R95p（95th percentile of daily precipitation）
    p95 = pr.quantile(0.95, dim="time", keep_attrs=True)
    r95p = indices._anuclim.prcptot(pr, p95)
    r95p.to_netcdf(f'E:/GEO/etccdi/qpm/r95p_{mod}_historical_{year}.nc')
    # 计算 R99p（99th percentile of daily precipitation）
    p99 = pr.quantile(0.99, dim="time", keep_attrs=True)
    r99p = indices._anuclim.prcptot(pr, p99)
    r99p.to_netcdf(f'E:/GEO/etccdi/qpm/r99p_{mod}_historical_{year}.nc')
    # 计算 r95ptot,r99ptot
    r95ptot = r95p / prcp
    r95ptot.to_netcdf(f'E:/GEO/etccdi/qpm/r95ptot_{mod}_historical_{year}.nc')
    r99ptot = r99p / prcp
    r99ptot.to_netcdf(f'E:/GEO/etccdi/qpm/r99ptot_{mod}_historical_{year}.nc')
    # 计算 SDII（平均降水日降水量）
    sdii = indices.daily_pr_intensity(pr)
    sdii.to_netcdf(f'E:/GEO/etccdi/qpm/sdii_{mod}_historical_{year}.nc')
    # 计算 Rx1day（连续 1 天内的最大日降水量）
    rx1day = indices.max_n_day_precipitation_amount(pr, window=1)
    rx1day.to_netcdf(f'E:/GEO/etccdi/qpm/rx1day_{mod}_historical_{year}.nc')
    # 计算 R10mm（连续 1 天中 10 mm 以上降水的次数）
    r10mm = ((pr > 10).groupby('time.dayofyear').sum(dim='time') > 0).sum(dim='dayofyear')
    # days_over_thresh = indices.days_over_precip_thresh(precip, thresh=20)
    r10mm.to_netcdf(f'E:/GEO/etccdi/qpm/r10mm_{mod}_historical_{year}.nc')
    # 计算 R20mm（连续 1 天中 20 mm 以上降水的次数）
    r20mm = ((pr > 20).groupby('time.dayofyear').sum(dim='time') > 0).sum(dim='dayofyear')
    #days_over_thresh = indices.days_over_precip_thresh(precip, thresh=20)
    r20mm.to_netcdf(f'E:/GEO/etccdi/qpm/r20mm_{mod}_historical_{year}.nc')

def main():
    total_tasks =len(names) * (1988 - 1961)
    with tqdm(total=total_tasks, desc="处理中") as pbar:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for mod in names:
                for year in range(1988, 2015):
                    futures.append(executor.submit(process_data, mod, year))
            for future in concurrent.futures.as_completed(futures):
                future.result()  # 等待任务完成，不处理返回值
                pbar.update(1)

if __name__ == '__main__':
    main()
