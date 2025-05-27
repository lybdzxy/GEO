from eofs.standard import Eof
import numpy as np
import xarray as xr
import pandas as pd

f = xr.open_dataset('25m.nc')
print(f)
pre = np.array(f['traj_path'])

print(f['traj_path'].dims)
print(pre.shape)
lat = f['lat']
lon = f['lon']
#计算纬度权重
lat = np.array(lat)
coslat = np.cos(np.deg2rad(lat))
wgts = np.sqrt(coslat)[..., np.newaxis]
#创建EOF分解器
solver = Eof(pre, weights=wgts)
#获取前三个模态，获取对应的PC序列和解释方差
eof = solver.eofsAsCorrelation(neofs=10)
pc = solver.pcs(npcs=10, pcscaling=1)
var = solver.varianceFraction()

dataset = xr.Dataset({
    'eof': (('mode', 'lat', 'lon'), eof),
}, coords={'lat': lat, 'lon': lon})
print(dataset)
# 保存为 NetCDF 文件
dataset.to_netcdf('25m_eof.nc')
# 保存 PC 序列（主成分）到 CSV
# 假设 pc.shape 为 (time, mode)
# 加入时间索引
pc_df = pd.DataFrame(pc, columns=[f'PC{i+1}' for i in range(pc.shape[1])])
pc_df['year'] = f['year'].values
pc_df.to_csv('25m_pc.csv', index=False)

# 保存每个模态的解释方差（百分比）
var_df = pd.DataFrame({
    'Mode': [f'Mode{i+1}' for i in range(len(var))],
    'VarianceFraction': var  # 不要加 .values
})
var_df.to_csv('25m_variance.csv', index=False)