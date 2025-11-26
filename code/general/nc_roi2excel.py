import pandas as pd
import xarray as xr
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import geopandas as gpd
import salem

index_values = ['r95p', 'r99p', 'rx1day', 'prcptot', 'sdii']
lons = np.arange(72.25, 136.25, 0.5)
lats = np.arange(53.75, 17.75, -0.5)
shp_dir = 'E:/GEO/result/geodata/ecm_el.shp'
shp = gpd.read_file(shp_dir)

def read_nc_file(index):
    datasets = []
    for year in range(1961, 2015):
        try:
            file_observed = f'E:/GEO/etccdi/{index}{year}.nc'
            dataset_observed = xr.open_dataset(file_observed)

            # 裁剪数据集
            dataset_observed = dataset_observed.salem.roi(shape=shp)
            datasets.append(dataset_observed)
        except Exception as e:
            print(f"Error processing {index} {year}: {e}")

    if not datasets:
        raise ValueError(f"No valid data found for index {index}")

    # 合并所有年份的数据集
    combined_dataset = xr.concat(datasets, dim='time')
    # 计算多年均值
    mean_dataset = combined_dataset.mean(dim='time')
    mean_dataset.to_netcdf(f'E:/GEO/{index}_mean.nc')
    return mean_dataset

def process_nc2excel(args):
    lon, lat, dataset = args
    try:
        pre = dataset['pre'].sel(longitude=lon, latitude=lat)
    except KeyError:
        return None
    if any(np.isnan(val) for val in [pre]):
        return None
    return lon, lat, pre.values.item()

if __name__ == '__main__':
    for index in index_values:
        # 加载数据集
        dataset = read_nc_file(index)

        # 创建进程池
        with ProcessPoolExecutor() as executor:
            lon_lat_list = [(lon, lat, dataset) for lon in lons for lat in lats]
            # 并行处理任务
            results = list(executor.map(process_nc2excel, lon_lat_list))

        # 处理结果
        result = []
        for result_item in results:
            if result_item is None:
                continue
            lon, lat, pre = result_item
            result.append((lon, lat, pre))

        # 调试输出
        print(f"Processed {index}: {len(result)} results")

        if result:
            # 将结果写入Excel文件
            df = pd.DataFrame(result, columns=['Lon', 'Lat', 'pre'])
            excel_file = f'E:/GEO/etccdi/{index}_mean.xlsx'
            df.to_excel(excel_file, index=False)
            print(f"Written to {excel_file}")
        else:
            print(f"No valid results for {index}")
