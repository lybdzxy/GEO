import xarray as xr
import numpy as np
import geopandas as gpd
import pandas as pd
import salem
import concurrent.futures
from tqdm import tqdm
from lmoments3 import distr


def fit_distribution(index, district_name):
    district_path = f'E:/GEO/result/pca/{district_name}.shp'
    district = gpd.read_file(district_path)
    totest = []
    for year in range(1961, 2015):
        data_path = f'E:/GEO/etccdi/{index}{year}.nc'
        data = xr.open_dataset(data_path)
        data = data.transpose('time', 'latitude', 'longitude')
        data_distr = data.salem.roi(shape=district)
        totest.extend(data_distr['pre'].values.flatten().tolist())

    # 将 totest 转换为 NumPy 数组
    totest = np.array(totest)
    # 删除 NaN 值
    totest = totest[~np.isnan(totest)]

    try:
        paras = distr.gev.lmom_fit(totest)

        loc = paras['loc']
        scale = paras['scale']
        shape = paras['c']

        return {'index': index, 'ssp': 'obs', 'district': district_name,
                'loc': loc, 'scale': scale, 'shape': shape}
    except (ZeroDivisionError, ValueError, RuntimeError):
        return None

def main():
    indices = ['rx1day', 'r99p', 'r95p']
    districts = ['NW', 'NE', 'SE']
    df = pd.DataFrame(columns=['index', 'ssp', 'district', 'loc', 'scale', 'shape'])

    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        futures = []
        for index in indices:
            for district_name in districts:
                futures.append(executor.submit(fit_distribution, index, district_name))

        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            result = future.result()
            results.append(result)

    # 将结果整理到 DataFrame 中
    for result in results:
        df = df.append(result, ignore_index=True)

    # 将 DataFrame 导出到 Excel 文件
    excel_file = 'E:/GEO/result/new/GEV_obs.xlsx'
    df.to_excel(excel_file, index=False)


if __name__ == '__main__':
    main()
