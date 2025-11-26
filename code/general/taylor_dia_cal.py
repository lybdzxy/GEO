import numpy as np
import xarray as xr
import pandas as pd

def calculate_rmse(observed, predicted):
    diff = predicted - observed
    squared_diff = np.square(diff)
    mean_squared_diff = np.mean(squared_diff)
    rmse = np.sqrt(mean_squared_diff)
    return rmse

def calculate_correlation(observed, predicted):
    correlation = np.corrcoef(observed, predicted)[0, 1]
    return correlation

def calculate_standard_deviation(data):
    std_dev = np.std(data)
    return std_dev

def calculate_bias(observed, predicted):
    bias = np.mean(predicted - observed)
    return bias

def calculate_variance_ratio_correlation(observed, predicted):
    obs_var = np.var(observed)
    pred_var = np.var(predicted)
    correlation = calculate_correlation(observed, predicted)
    variance_ratio = (pred_var / obs_var) / (correlation**2)
    return variance_ratio

def calculate_nse(observed, predicted):
    mean_observed = np.mean(observed)
    numerator = np.sum((predicted - observed) ** 2)
    denominator = np.sum((observed - mean_observed) ** 2)
    nse = 1 - (numerator / denominator)
    return nse

def calculate_kge(observed, predicted):
    mean_observed = np.mean(observed)
    mean_predicted = np.mean(predicted)
    std_observed = np.std(observed)
    std_predicted = np.std(predicted)
    corr = np.corrcoef(observed, predicted)[0, 1]
    kge = 1 - np.sqrt((corr - 1) ** 2 + (std_predicted / std_observed - 1) ** 2 + (mean_predicted / mean_observed - 1) ** 2)
    return kge

def calculate_modin(observed, predicted):
    numerator = np.sum(np.abs(predicted - observed))
    denominator = np.sum(np.abs(observed - np.mean(observed)) + np.abs(predicted - np.mean(observed)))
    modin = 1 - (numerator / denominator)
    return modin

# 创建空的 DataFrame 以保存指标
results_df = pd.DataFrame(columns=['Model', 'Index', 'Year', 'RMSE', 'Correlation', 'Std_Dev_Predicted', 'Bias', 'Var_Ratio_Corr', 'NSE', 'KGE', 'ModIn'])

names = ['EC-Earth3', 'MPI-ESM1-2-HR', 'EC-Earth3-Veg',
         'CMCC-ESM2', 'GFDL-ESM4', 'INM-CM4-8',
         'BCC-CSM2-MR', 'MRI-ESM2-0', 'CMCC-CM2-SR5',
         'TaiESM1', 'NorESM2-MM', 'INM-CM5-0']
indices = ['prcptot', 'r95p', 'r99p', 'r95ptot', 'r99ptot', 'sdii', 'rx1day', 'r10mm', 'r20mm']

for mod in names:
    for index in indices:
        for year in range(1988, 2015):
            # 文件路径
            file_observed = f'E:/GEO/etccdi/chm_pre/cut/{index}{year}.nc'
            file_predicted = f'E:/GEO/etccdi/qpm/cut/{index}_{mod}_historical_{year}.nc'

            # 打开 NetCDF 文件
            dataset_observed = xr.open_dataset(file_observed)
            dataset_predicted = xr.open_dataset(file_predicted)

            # 读取变量数据为数组
            observed = dataset_observed['pre'].values
            predicted = dataset_predicted['pre'].values

            # 去除NaN值
            observed = observed[~np.isnan(observed)]
            predicted = predicted[~np.isnan(predicted)]

            # 计算指标
            rmse_value = calculate_rmse(observed, predicted)
            correlation = calculate_correlation(observed, predicted)
            std_dev_predicted = calculate_standard_deviation(predicted)
            bias = calculate_bias(observed, predicted)
            variance_ratio_correlation = calculate_variance_ratio_correlation(observed, predicted)
            nse_value = calculate_nse(observed, predicted)
            kge_value = calculate_kge(observed, predicted)
            modin_value = calculate_modin(observed, predicted)

            # 添加到 DataFrame
            results_df = results_df.append({'Model': mod,
                                            'Index': index,
                                            'Year': year,
                                            'RMSE': rmse_value,
                                            'Correlation': correlation,
                                            'Std_Dev_Predicted': std_dev_predicted,
                                            'Bias': bias,
                                            'Var_Ratio_Corr': variance_ratio_correlation,
                                            'NSE': nse_value,
                                            'KGE': kge_value,
                                            'ModIn': modin_value}, ignore_index=True)

            # 关闭 NetCDF 文件
            dataset_observed.close()
            dataset_predicted.close()

# 保存结果到 Excel 文件
results_df.to_excel('E:/GEO/etccdi/qpm/model_indices_results.xlsx', index=False)
