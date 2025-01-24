import xarray as xr
import numpy as np
import geopandas as gpd
import pandas as pd
import salem
import concurrent.futures
from tqdm import tqdm


def process_data(index, ssp, district_name, year):
    district_path = f'E:/GEO/result/geodata/district/{district_name}.shp'
    district = gpd.read_file(district_path)
    data_path = f'E:/GEO/etccdi/{index}{year}.nc'
    data = xr.open_dataset(data_path)
    data = data.transpose('time', 'latitude', 'longitude')
    data_distr = data.salem.roi(shape=district)
    # 将数据集转换为数据数组
    data_array = data_distr.to_array()
    # 计算年度平均值
    yearly_mean = np.nanmean(data_array)  # 假设时间维度为 'time'
    return {'index': index, 'ssp': ssp, 'district': district_name, 'year': year, 'mean_value': yearly_mean}


if __name__ == '__main__':
    indices = ['prcptot', 'r95p', 'r99p', 'sdii', 'rx1day']
    ssps = ['obs']
    districts = ['E', 'N', 'NC', 'NE', 'S', 'SC', 'W']
    years = range(1961, 2015)

    # 创建一个空的列表用于存储结果
    result_list = []

    # 使用ProcessPoolExecutor并行处理数据
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for index in indices:
            for ssp in ssps:
                for district_name in districts:
                    for year in years:
                        futures.append(executor.submit(process_data, index, ssp, district_name, year))

        # 处理并获取结果
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Processing'):
            result = future.result()
            result_list.append(result)

    # 将结果转换为DataFrame
    result_df = pd.DataFrame(result_list)

    # 将结果保存为Excel文件
    for district_name in districts:
        for ssp in ssps:
            for index in indices:
                filtered_df = result_df[(result_df['index'] == index) & (result_df['ssp'] == ssp) & (
                            result_df['district'] == district_name)]
                excel_file_name = f'E:/GEO/result/ecm/{district_name}_{ssp}_{index}.xlsx'  # 指定Excel文件名
                filtered_df.to_excel(excel_file_name, index=False)
