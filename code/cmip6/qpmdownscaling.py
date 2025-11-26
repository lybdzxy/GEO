import xarray as xr
import numpy as np
import pandas as pd

# 打开NetCDF文件
his = xr.open_dataset('E:/GEO/test/interpolated_his.nc')
ssp = xr.open_dataset('E:/GEO/test/interpolated_ssp.nc')
lon_values = np.arange(72.25, 136.25, 0.5)
lat_values = np.arange(53.75, 17.75, -0.5)

results={}

for lon in lon_values:
    for lat in lat_values:
        # 从xarray数据集中获取数据
        his_pr = his['pr'].sel(longitude=lon, latitude=lat)
        ssp_pr = ssp['pr'].sel(longitude=lon, latitude=lat)
        # 对数据进行排序
        sorted_his = np.sort(his_pr)
        sorted_ssp = np.sort(ssp_pr)
        qp = np.divide(sorted_his, sorted_ssp)
        results[(lon, lat)] = {'qp': qp}

# 保存为一个 NetCDF 文件
result_df = {(lon, lat, rank): result_data['qp'][rank] for (lon, lat), result_data in results.items() for rank in range(365)}
result_df = pd.DataFrame.from_dict(result_df, orient='index', columns=['qp'])
result_df.index = pd.MultiIndex.from_tuples(result_df.index, names=['lon', 'lat', 'rank'])
result_df.to_xarray().to_netcdf('E:/GEO/test/qp_results.nc')




obs = xr.open_dataset('E:/GEO/test/CHM_PRE1961.nc')
lon_values = np.arange(72.25, 136.25, 0.5)
lat_values = np.arange(53.75, 17.75, -0.5)

results={}

for lon in lon_values:
    for lat in lat_values:
        # 从xarray数据集中获取数据
        obs_pr = obs['pre'].sel(longitude=lon, latitude=lat)
        # 对数据进行排序
        sorted_indices_obs = np.argsort(obs_pr)
        sorted_obs_time = obs_pr.time.values[sorted_indices_obs]
        sorted_obs = obs_pr.values[sorted_indices_obs]
        results[(lon, lat)] = {'pr': sorted_obs, 'sorted_time': sorted_obs_time}

# 保存为一个 NetCDF 文件
result_df = {(lon, lat, rank): result_data['pr'][rank] for (lon, lat), result_data in results.items() for rank in range(365)}
result_df = pd.DataFrame.from_dict(result_df, orient='index', columns=['pr'])
result_df.index = pd.MultiIndex.from_tuples(result_df.index, names=['lon', 'lat', 'rank'])
result_xr = result_df.to_xarray()
result_xr['sorted_time'] = xr.DataArray(np.concatenate([result_data['sorted_time'] for result_data in results.values()]),
                                              dims=('index'))
result_xr.to_netcdf('E:/GEO/test/obs_rank.nc')




obs_rank = xr.open_dataset('E:/GEO/test/obs_rank.nc')
qp_rank = xr.open_dataset('E:/GEO/test/qp_results.nc')

downed = obs_rank['pr']*qp_rank['qp']

sorted_time_obs = obs_rank['sorted_time']

downed_sorted = downed.sortby(sorted_time_obs)

downed_sorted.to_netcdf('E:/GEO/test/downed_sorted.nc')
