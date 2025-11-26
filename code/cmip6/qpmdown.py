import xarray as xr
import numpy as np
import pandas as pd

obs_rank = xr.open_dataset('E:/GEO/test/obs_rank.nc')
qp_rank = xr.open_dataset('E:/GEO/test/qp_results.nc')

downed = obs_rank['pr']*qp_rank['qp']
# 将大于某个值和小于某个值的部分设为 NaN
threshold_min = 0  # 小于这个值的部分将被设为 NaN
threshold_max = 1000  # 大于这个值的部分将被设为 NaN

downed_filtered = downed.where((downed >= threshold_min) & (downed <= threshold_max))

# 将结果保存到文件
downed_filtered.to_netcdf('E:/GEO/test/downed_filtered.nc')
