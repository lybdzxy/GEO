import xarray as xr
import numpy as np
from scipy.stats import genextreme
from scipy import stats
import geopandas as gpd
import pandas as pd
import salem
import concurrent.futures
from tqdm import tqdm


def fit_distribution(index, ssp, district_name):
    district_path = f'E:/GEO/geodata/{district_name}.shp'
    district = gpd.read_file(district_path)
    ori = []
    for year in range(1961, 2015):
        data_path = f'E:/GEO/etccdi/{index}{year}.nc'
        data = xr.open_dataset(data_path)
        data = data.transpose('time', 'latitude', 'longitude')
        data_distr = data.salem.roi(shape=district)
        ori.extend(data_distr['pre'].values.flatten().tolist())

    # 将 totest 转换为 NumPy 数组
    ori = np.array(ori)

    # 删除 NaN 值
    ori = ori[~np.isnan(ori)]

    totest = []
    for year in range(2015, 2068):
        data_path = f'E:/GEO/etccdi/qpm/mme/cut/{index}_{ssp}_{year}.nc'
        data = xr.open_dataset(data_path)
        data = data.transpose('time', 'lat', 'lon')
        data_distr = data.salem.roi(shape=district)
        totest.extend(data_distr['__xarray_dataarray_variable__'].values.flatten().tolist())

    # 将 totest 转换为 NumPy 数组
    totest = np.array(totest)

    # 删除 NaN 值
    totest = totest[~np.isnan(totest)]
    statistic, p_value = stats.ks_2samp(ori, totest)
    return {'index': index, 'ssp': ssp, 'district': district_name,
            'statistic': statistic, 'p_value': p_value}


def main():
    indices = ['rx1day', 'r99p', 'r95p']
    ssps = ['126', '245', '370', '585']
    districts = ['CC', 'SWC', 'NWC', 'SC', 'EC', 'NC', 'NEC']
    df = pd.DataFrame(columns=['index', 'ssp', 'district', 'statistic', 'p_value'])

    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        futures = []
        for index in indices:
            for ssp in ssps:
                for district_name in districts:
                    futures.append(executor.submit(fit_distribution, index, ssp, district_name))

        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            result = future.result()
            results.append(result)

    # 将结果整理到 DataFrame 中
    for result in results:
        df = df.append(result, ignore_index=True)

    # 将 DataFrame 导出到 Excel 文件
    excel_file = 'E:/GEO/result/qpm/gev_ks.xlsx'
    df.to_excel(excel_file, index=False)


if __name__ == '__main__':
    main()
