import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm

for year in tqdm(range(1961,2023)):
    obs = xr.open_dataset(f'E:/GEO/chm_pre/CHM_PRE_0.5dg_19612022.nc/CHM_PRE{year}.nc')
    lon_values = np.arange(72.25, 136.25, 0.5)
    lat_values = np.arange(53.75, 17.75, -0.5)

    results={}

    for lon in lon_values:
        for lat in lat_values:
            # 从xarray数据集中获取数据
            obs_pr = obs['pre'].sel(longitude=lon, latitude=lat)
            # 对数据进行排序
            sorted_indices_obs = np.argsort(obs_pr)
            sorted_obs = obs_pr.values[sorted_indices_obs]
            if len(sorted_obs) == 366:  # 如果有366个数据，删除第一个
                sorted_obs = np.delete(sorted_obs, 0)
            results[(lon, lat)] = {'pr': sorted_obs}

    # 创建时间变量
    time_index = pd.date_range(start='2000-01-01', periods=365)
    # 保存为一个 NetCDF 文件
    result_df = {(lon, lat, time): result_data['pr'][time_idx] for (lon, lat), result_data in results.items() for time_idx, time in enumerate(time_index)}
    result_df = pd.DataFrame.from_dict(result_df, orient='index', columns=['pr'])
    result_df.index = pd.MultiIndex.from_tuples(result_df.index, names=['lon', 'lat', 'time'])
    result_xr = result_df.to_xarray().to_netcdf(f'E:/GEO/down/qpm/mid/obs_rank{year}.nc')
