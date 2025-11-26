import xarray as xr
import concurrent.futures
from tqdm import tqdm

names = ['EC-Earth3', 'MPI-ESM1-2-HR', 'EC-Earth3-Veg',
         'CMCC-ESM2', 'GFDL-ESM4', 'INM-CM4-8',
         'BCC-CSM2-MR', 'MRI-ESM2-0', 'CMCC-CM2-SR5',
         'TaiESM1', 'NorESM2-MM', 'INM-CM5-0']

def process_data(mod, year):
    obs_rank = xr.open_dataset(f'E:/GEO/down/qpm/mid/obs_rank{year}.nc')
    qp_rank = xr.open_dataset(f'E:/GEO/down/qpm/mid/qp_{mod}_his_{year}.nc')
    ssp_year = year + 27
    downed = obs_rank['pr'] * qp_rank['qp']
    threshold_min = 0
    threshold_max = 1000
    downed_filtered = downed.where((downed >= threshold_min) & (downed <= threshold_max))
    downed_filtered.to_netcdf(f'E:/GEO/down/qpm/downed_{mod}_historical_{ssp_year}.nc')

def main():
    total_tasks =len(names) * (1988 - 1961)
    with tqdm(total=total_tasks, desc="处理中") as pbar:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for mod in names:
                for year in range(1961, 1988):
                    futures.append(executor.submit(process_data, mod, year))
            for future in concurrent.futures.as_completed(futures):
                future.result()  # 等待任务完成，不处理返回值
                pbar.update(1)

if __name__ == '__main__':
    main()
