import numpy as np
import xarray as xr
import pandas as pd
import salem
import geopandas as gpd

def calculate_rmse(observed, predicted):
    return np.sqrt(np.mean(np.square(predicted - observed)))

def calculate_correlation(observed, predicted):
    return np.corrcoef(observed, predicted)[0, 1]

def calculate_standard_deviation(data):
    return np.std(data)

def calculate_bias(observed, predicted):
    return np.mean(predicted - observed)

def calculate_variance_ratio_correlation(observed, predicted):
    obs_var = np.var(observed)
    pred_var = np.var(predicted)
    correlation = calculate_correlation(observed, predicted)
    return (pred_var / obs_var) / (correlation**2)

def calculate_nse(observed, predicted):
    mean_observed = np.mean(observed)
    return 1 - (np.sum((predicted - observed) ** 2) / np.sum((observed - mean_observed) ** 2))

def calculate_kge(observed, predicted):
    mean_observed = np.mean(observed)
    mean_predicted = np.mean(predicted)
    std_observed = np.std(observed)
    std_predicted = np.std(predicted)
    corr = np.corrcoef(observed, predicted)[0, 1]
    return 1 - np.sqrt((corr - 1) ** 2 + (std_predicted / std_observed - 1) ** 2 + (mean_predicted / mean_observed - 1) ** 2)

def calculate_modin(observed, predicted):
    numerator = np.sum(np.abs(predicted - observed))
    denominator = np.sum(np.abs(observed - np.mean(observed)) + np.abs(predicted - np.mean(observed)))
    return 1 - (numerator / denominator)

def calculate_taylor_skill_score(observed, predicted):
    correlation = calculate_correlation(observed, predicted)
    std_observed = calculate_standard_deviation(observed)
    std_predicted = calculate_standard_deviation(predicted)
    bias = calculate_bias(observed, predicted)
    return 1 - (std_predicted - std_observed)**2 / std_observed**2 - (bias / std_observed)**2

def calculate_interannual_variability_skill_score(observed, predicted):
    mean_observed = np.mean(observed)
    return 1 - (np.sum((predicted - observed) ** 2) / np.sum((observed - mean_observed) ** 2))

def calculate_nrmse(observed, predicted):
    rmse_value = calculate_rmse(observed, predicted)
    return rmse_value / (np.max(observed) - np.min(observed))

def remove_nan_values(observed, predicted):
    mask = ~np.isnan(observed) & ~np.isnan(predicted)
    return observed[mask], predicted[mask]

# 创建空的 DataFrame 以保存指标
results_df = pd.DataFrame(columns=['Model', 'Index', 'Year', 'Correlation', 'NSE', 'KGE', 'Taylor_Skill_Score', 'Interannual_Variability_Skill_Score', 'NRMSE'])
all_results = []

names = ['EC-Earth3', 'MPI-ESM1-2-HR', 'EC-Earth3-Veg', 'CMCC-ESM2', 'GFDL-ESM4', 'INM-CM4-8', 'BCC-CSM2-MR', 'MRI-ESM2-0', 'CMCC-CM2-SR5', 'TaiESM1', 'NorESM2-MM', 'INM-CM5-0']
indices = ['prcptot', 'r95p', 'r99p', 'sdii', 'rx1day']
shp_dir = 'E:/GEO/result/geodata/ecm_el.shp'
shp = gpd.read_file(shp_dir)

for mod in names:
    for index in indices:
        observed_all_years = []
        predicted_all_years = []
        for year in range(1988, 2015):
            try:
                file_observed = f'E:/GEO/etccdi/chm_pre/cut/{index}{year}.nc'
                file_predicted = f'E:/GEO/etccdi/qpm/cut/{index}_{mod}_historical_{year}.nc'

                dataset_observed = xr.open_dataset(file_observed)
                dataset_predicted = xr.open_dataset(file_predicted)

                dataset_observed = dataset_observed.salem.roi(shape=shp)
                dataset_predicted = dataset_predicted.salem.roi(shape=shp)

                observed = dataset_observed['pre'].values
                predicted = dataset_predicted['pre'].values

                observed_all_years.append(observed)
                predicted_all_years.append(predicted)

                dataset_observed.close()
                dataset_predicted.close()
            except Exception as e:
                print(f"Error processing {mod} {index} {year}: {e}")

        # 计算多年均值
        observed_mean = np.mean(observed_all_years, axis=0)
        predicted_mean = np.mean(predicted_all_years, axis=0)

        # 去除NaN值
        observed_mean, predicted_mean = remove_nan_values(observed_mean, predicted_mean)

        # 计算指标
        rmse_value = calculate_rmse(observed_mean, predicted_mean)
        correlation = calculate_correlation(observed_mean, predicted_mean)
        std_dev_predicted = calculate_standard_deviation(predicted_mean)
        bias = calculate_bias(observed_mean, predicted_mean)
        variance_ratio_correlation = calculate_variance_ratio_correlation(observed_mean, predicted_mean)
        nse_value = calculate_nse(observed_mean, predicted_mean)
        kge_value = calculate_kge(observed_mean, predicted_mean)
        modin_value = calculate_modin(observed_mean, predicted_mean)
        taylor_skill_score = calculate_taylor_skill_score(observed_mean, predicted_mean)
        interannual_variability_skill_score = calculate_interannual_variability_skill_score(observed_mean, predicted_mean)
        nrmse_value = calculate_nrmse(observed_mean, predicted_mean)

        result = {'Model': mod,
                  'Index': index,
                  'Year': '1988-2014',  # 多年均值，年份范围
                  'Correlation': correlation,
                  'NSE': nse_value,
                  'KGE': kge_value,
                  'Taylor_Skill_Score': taylor_skill_score,
                  'Interannual_Variability_Skill_Score': interannual_variability_skill_score,
                  'NRMSE': nrmse_value}

        all_results.append(result)

# 保存结果到Excel文件
results_df = pd.DataFrame(all_results)
results_df.to_excel('E:/GEO/etccdi/qpm/multi_year_result.xlsx', index=False)
