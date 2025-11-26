import xarray as xr
import geopandas as gpd
import salem
from tqdm import tqdm
import concurrent.futures

# 定义处理函数
def process_data(index, mod, year):
    shp_dir = f'E:/GEO/geodata/CNTOT.shp'
    ming_shp2 = gpd.read_file(shp_dir)
    nc_path = f'E:/GEO/etccdi/qpm/tr/{index}_{mod}_historical_{year}.nc'
    output_nc_path = f'E:/GEO/etccdi/qpm/cut/{index}_{mod}_historical_{year}.nc'
    ds = xr.open_dataset(nc_path)
    ming = ds.salem.roi(shape=ming_shp2)
    variable_names = ming.variables.keys()
    if '__xarray_dataarray_variable__' in variable_names:
        variable_name = '__xarray_dataarray_variable__'
    elif 'pre' in variable_names:
        variable_name = 'pre'
    ming['pre'] = ming[variable_name].astype('float64')
    ming.to_netcdf(output_nc_path)

def main():
    names = ['EC-Earth3', 'MPI-ESM1-2-HR', 'EC-Earth3-Veg',
             'CMCC-ESM2', 'GFDL-ESM4', 'INM-CM4-8',
             'BCC-CSM2-MR', 'MRI-ESM2-0', 'CMCC-CM2-SR5',
             'TaiESM1', 'NorESM2-MM', 'INM-CM5-0']
    indices = ['prcptot', 'r95p', 'r99p', 'r95ptot', 'r99ptot', 'sdii', 'rx1day', 'r10mm', 'r20mm']

    total_tasks = len(indices) * len(names) * (2015 - 1988)
    with tqdm(total=total_tasks, desc="处理中") as pbar:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for index in indices:
                for mod in names:
                    for year in range(1988, 2015):
                        futures.append(executor.submit(process_data, index, mod, year))
            for future in concurrent.futures.as_completed(futures):
                future.result()  # 等待任务完成，不处理返回值
                pbar.update(1)

if __name__ == '__main__':
    main()
