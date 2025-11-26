import xarray as xr
import geopandas as gpd
import salem
from tqdm import tqdm
import concurrent.futures

# 定义处理函数
def process_data(index, ssp, year):
    shp_dir = f'E:/GEO/geodata/CNTOT.shp'
    ming_shp2 = gpd.read_file(shp_dir)
    nc_path = f'E:/GEO/etccdi/qpm/mme/{index}_{ssp}_{year}.nc'
    output_nc_path = f'E:/GEO/etccdi/qpm/mme/cut/{index}_{ssp}_{year}.nc'
    ds = xr.open_dataset(nc_path)
    if 'time' in ds.dims:
        ds = ds.transpose('time', 'lat', 'lon')
    else:
        ds = ds.transpose('lat', 'lon')

    # 对空间维度进行裁剪
    ming = ds.salem.roi(shape=ming_shp2)
    ming = ming.where(~((ds.lon >= 120.25) & (ds.lon <= 121.75) & (ds.lat >= 22.25) & (ds.lat <= 24.75)),
                      other=float('nan'))
    ming.to_netcdf(output_nc_path)

def main():
    indices = ['prcptot', 'r95p', 'r99p', 'r95ptot', 'r99ptot', 'sdii', 'rx1day', 'r10mm', 'r20mm']
    ssps = ['126', '245', '370', '585']
    years = range(2015,2069)
    total_tasks = len(indices) * len(ssps) *len(years)
    with tqdm(total=total_tasks, desc="处理中") as pbar:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for index in indices:
                for ssp in ssps:
                    for year in years:
                        futures.append(executor.submit(process_data, index, ssp, year))
            for future in concurrent.futures.as_completed(futures):
                future.result()  # 等待任务完成，不处理返回值
                pbar.update(1)

if __name__ == '__main__':
    main()
