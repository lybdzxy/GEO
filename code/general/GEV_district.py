import xarray as xr
import numpy as np
from scipy.stats import genextreme,zscore
from scipy.optimize import curve_fit
import geopandas as gpd
import pandas as pd
import salem
import concurrent.futures
from tqdm import tqdm


def fit_distribution(index, district_name):
    district_path = f'E:/GEO/geodata/{district_name}.shp'
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
    '''# 进行zscore
    totest_zscored = zscore(totest)

    # 剔除z值大于2的部分
    totest_filtered = [x for i, x in enumerate(totest) if totest_zscored[i] <= 2.5]'''
    initial_guess = [np.mean(totest), np.std(totest), 0]

    # 创建概率密度函数值的列表
    x_values = np.linspace(min(totest), max(totest), len(totest))
    y_values = genextreme.pdf(x_values, initial_guess[2], loc=initial_guess[0], scale=initial_guess[1])

    # 拟合广义极值分布
    params, cov = curve_fit(lambda x, loc, scale, shape: genextreme.pdf(x, shape, loc=loc, scale=scale),
                            x_values, y_values, p0=initial_guess)  # 增加maxfev参数

    return {'index': index, 'ssp': 'obs', 'district': district_name,
            'loc': params[0], 'scale': params[1], 'shape': params[2]}


def main():
    indices = ['rx1day', 'r99p', 'r95p']
    '''ssps = ['126', '245', '370', '585']'''
    districts = ['CC', 'SWC', 'NWC', 'SC', 'EC', 'NC', 'NEC']
    df = pd.DataFrame(columns=['index', 'ssp', 'district', 'loc', 'scale', 'shape'])

    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        futures = []
        for index in indices:
            '''for ssp in ssps:'''
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
    excel_file = 'E:/GEO/result/qpm/obs_params.xlsx'
    df.to_excel(excel_file, index=False)


if __name__ == '__main__':
    main()
