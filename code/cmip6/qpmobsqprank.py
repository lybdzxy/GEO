import concurrent.futures
import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm

names = ['EC-Earth3', 'MPI-ESM1-2-HR', 'EC-Earth3-Veg',
         'CMCC-ESM2', 'GFDL-ESM4', 'INM-CM4-8',
         'BCC-CSM2-MR', 'MRI-ESM2-0', 'CMCC-CM2-SR5',
         'TaiESM1', 'NorESM2-MM', 'CESM2-WACCM', 'INM-CM5-0']

def process_data(mod, year):
    results = {}
    his = xr.open_dataset(f'E:/GEO/CMIP6/interpolated_pr_day_{mod}_historical_{year}.nc')
    ssp_year = year + 27
    ssp = xr.open_dataset(f'E:/GEO/CMIP6/interpolated_pr_day_{mod}_historical_{ssp_year}.nc')

    lon_values = np.arange(72.25, 136.25, 0.5)
    lat_values = np.arange(53.75, 17.75, -0.5)

    for lon in lon_values:
        for lat in lat_values:
            his_pr = his['pr'].sel(longitude=lon, latitude=lat)
            ssp_pr = ssp['pr'].sel(longitude=lon, latitude=lat)
            sorted_his = np.sort(his_pr)
            if len(sorted_his) == 366:
                sorted_his = np.delete(sorted_his, 0)
            sorted_ssp = np.sort(ssp_pr)
            if len(sorted_ssp) == 366:
                sorted_ssp = np.delete(sorted_ssp, 0)
            qp = np.divide(sorted_ssp, sorted_his)
            results[(lon, lat)] = {'qp': qp}

    time_index = pd.date_range(start='2000-01-01', periods=365)

    result_df = {(lon, lat, time): result_data['qp'][time_idx] for (lon, lat), result_data in results.items() for time_idx, time in enumerate(time_index)}
    result_df = pd.DataFrame.from_dict(result_df, orient='index', columns=['qp'])
    result_df.index = pd.MultiIndex.from_tuples(result_df.index, names=['lon', 'lat', 'time'])
    result_xr = result_df.to_xarray().to_netcdf(f'E:/GEO/down/qpm/mid/qp_{mod}_his_{year}.nc')

def main():
    total_tasks = len(names) * (1988 - 1961)
    with tqdm(total=total_tasks, desc="处理中") as pbar:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for mod in names:
                for year in range(1961, 1988):
                    futures.append(executor.submit(process_data, mod, year))
            for future in concurrent.futures.as_completed(futures):
                future.result()  # 等待任务完成，不处理返回值
                pbar.update(1)
            # 等待所有任务完成
            concurrent.futures.wait(futures)

if __name__ == '__main__':
    main()
