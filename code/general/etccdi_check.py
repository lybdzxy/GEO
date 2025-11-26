import os
import xarray as xr
import numpy as np

def check_data_shapes(dataset_observed, dataset_predicted, path_predicted):
    observed = dataset_observed['pre'].values
    predicted = dataset_predicted['pre'].values
    observed = observed[~np.isnan(observed)]
    predicted = predicted[~np.isnan(predicted)]
    if observed.shape != predicted.shape:
        print(f"Shapes mismatch for {path_predicted}: Observed shape {observed.shape}, Predicted shape {predicted.shape}")

names = ['EC-Earth3', 'MPI-ESM1-2-HR', 'EC-Earth3-Veg',
         'CMCC-ESM2', 'GFDL-ESM4', 'INM-CM4-8',
         'BCC-CSM2-MR', 'MRI-ESM2-0', 'CMCC-CM2-SR5',
         'TaiESM1', 'NorESM2-MM', 'INM-CM5-0']
indices = ['prcptot', 'r95p', 'r99p', 'r95ptot', 'r99ptot', 'sdii', 'rx1day', 'r10mm', 'r20mm']

for mod in names:
    for index in indices:
        for year in range(1988, 2015):
            # 文件路径
            path_observed = f'E:/GEO/etccdi/chm_pre/cut/{index}{year}.nc'
            path_predicted = f'E:/GEO/etccdi/qpm/cut/{index}_{mod}_historical_{year}.nc'

            if not os.path.exists(path_predicted):
                print(f"File {path_predicted} not found.")
                continue

            dataset_observed = xr.open_dataset(path_observed)
            dataset_predicted = xr.open_dataset(path_predicted)

            check_data_shapes(dataset_observed, dataset_predicted, path_predicted)

            dataset_observed.close()
            dataset_predicted.close()
