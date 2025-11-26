import xarray as xr
import geopandas as gpd
import salem
from tqdm import tqdm
import concurrent.futures

# 定义处理函数
def process_data(index, ssp, name):
    shp_dir = f'E:/GEO/geodata/CNTOT.shp'
    ming_shp2 = gpd.read_file(shp_dir)
    nc_path = f'E:/GEO/result/qpm/{ssp}{index}_{name}_zs.nc'
    output_nc_path = f'E:/GEO/result/qpm/cut/{ssp}{index}_{name}_zs.nc'
    ds = xr.open_dataset(nc_path)
    # 将 level_0 重命名为 lon，level_1 重命名为 lat
    ds = ds.rename({'level_0': 'lon', 'level_1': 'lat'})

    # 裁剪之前重新排列数据集的坐标轴
    ds = ds.transpose('lat', 'lon')

    # 对空间维度进行裁剪
    ming = ds.salem.roi(shape=ming_shp2)
    ming = ming.where(~((ds.lon >= 120.25) & (ds.lon <= 121.75) & (ds.lat >= 22.25) & (ds.lat <= 24.75)),
                      other=float('nan'))
    ming.to_netcdf(output_nc_path)

def main():
    indices = ['r95p', 'r99p','rx1day']
    ssps = ['126', '245', '370', '585']
    names = ['precipitation_3_year', 'precipitation_10_year', 'precipitation_20_year', 'precipitation_50_year', 'precipitation_100_year']
    total_tasks = len(indices) * len(ssps)
    with tqdm(total=total_tasks, desc="处理中") as pbar:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for index in indices:
                for ssp in ssps:
                    for name in names:
                        futures.append(executor.submit(process_data, index, ssp, name))
            for future in concurrent.futures.as_completed(futures):
                future.result()  # 等待任务完成，不处理返回值
                pbar.update(1)

if __name__ == '__main__':
    main()
