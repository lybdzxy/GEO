import xarray as xr
import numpy as np

# 打开nc文件
ds = xr.open_dataset('traj_paths_5deg.nc')

# 假设变量名为'traj_path'，你需要根据实际情况修改
frequency = ds['traj_path']

# 遍历所有格点
for i in range(frequency.shape[1]):  # 经度维度
    for j in range(frequency.shape[2]):  # 纬度维度
        # 获取某个格点所有年份的频数数据
        data = frequency[:, i, j].values

        # 计算数据中为0的数量
        zero_count = np.sum(data == 0)
        total_count = len(data)

        # 如果0的数量超过一半，将该格点所有年份的数据设为NaN
        if zero_count > total_count / 2:
            frequency[:, i, j] = np.nan

# 保存修改后的nc文件
ds.to_netcdf('5m.nc')

# 关闭文件
ds.close()
