from eofs.standard import Eof
import numpy as np
import xarray as xr

f = xr.open_dataset('E:/GEO/eof/CHM_PRE_PRCPTOT.nc')
pre = np.array(f['pre'])
lat = f['latitude']
lon = f['longitude']
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
    'eof': (('mode', 'latitude', 'longitude'), eof),
}, coords={'lat': lat, 'lon': lon})
print(dataset)
# 保存为 NetCDF 文件
dataset.to_netcdf('E:/GEO/eof/eof.nc')