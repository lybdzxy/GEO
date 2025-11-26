import pandas as pd
import xarray as xr
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import geopandas as gpd
import salem

ssp_values = ['obs']  # 将字符串改为整数
index_values = ['r95p', 'r99p', 'rx1day']  # 保持字符串不变
lons = np.arange(72.25, 136.25, 0.5)
lats = np.arange(53.75, 17.75, -0.5)
shp_dir = 'E:/GEO/result/geodata/ecm_el.shp'
shp = gpd.read_file(shp_dir)

def read_nc_file(ssp, index):
    data_path = f'E:/GEO/result/qpm/cut/{ssp}{index}_lom_GEV.nc'
    dataset = xr.open_dataset(data_path)
    dataset = dataset.salem.roi(shape=shp)
    return dataset


def process_nc2excel(lon_lat, dataset):
    lon, lat = lon_lat
    try:
        loc = dataset['loc'].sel(lon=lon, lat=lat)
        scale = dataset['scale'].sel(lon=lon, lat=lat)
        shape = dataset['shape'].sel(lon=lon, lat=lat)
    except KeyError:
        return None
    if any(np.isnan(val) for val in [loc, scale, shape]):
        return None
    return lon, lat, loc.values.item(), scale.values.item(), shape.values.item()


if __name__ == '__main__':
    for ssp in ssp_values:
        for index in index_values:
            # 加载数据集
            dataset = read_nc_file(ssp, index)

            # 创建进程池
            with ProcessPoolExecutor() as executor:
                # 使用Manager来创建共享的lons和lats
                manager = multiprocessing.Manager()
                lon_lat_list = [(lon, lat) for lon in lons for lat in lats]
                # 并行处理任务
                results = executor.map(process_nc2excel, lon_lat_list, [dataset] * len(lon_lat_list))

            # 处理结果
            result = {}
            for result_item in results:
                if result_item is None:
                    continue
                lon, lat, loc, scale, shape = result_item
                result[(lon, lat)] = (loc, scale, shape)

            # 将结果写入Excel文件
            df = pd.DataFrame(result).T.reset_index()
            df.columns = ['Lon', 'Lat', 'Loc', 'Scale', 'Shape']
            excel_file = f'E:/GEO/result/ecm/{ssp}{index}_lom_GEV.xlsx'
            df.to_excel(excel_file, index=False)
