from eofs.standard import Eof
import numpy as np
import xarray as xr
import pandas as pd

'modified_file.nc'
'traj_paths_1deg'
# 打开数据
f = xr.open_dataset('traj_paths_1deg.nc')
pre = f['traj_path']
lat = f['lat']
lon = f['lon']

# 区域定义（按经纬度范围划分）
regions = {
    'eurasia': {'lat_min': 40, 'lat_max': 60, 'lon_min': 75, 'lon_max': 105},
    'north_america': {'lat_min': 35, 'lat_max': 55, 'lon_min': -125, 'lon_max': -100},
    'north_pacific': {'lat_min': 30, 'lat_max': 50, 'lon_min': 165, 'lon_max': -130},  # 需处理跨越180经线
    'north_atlantic': {'lat_min': 30, 'lat_max': 50, 'lon_min': -60, 'lon_max': -15},
    'greenland':{'lat_min': 65, 'lat_max': 80, 'lon_min': -50, 'lon_max': -25}
}

# 区域数据裁剪函数
def subset_region(data, lat, lon, region):
    lat_cond = (lat >= region['lat_min']) & (lat <= region['lat_max'])
    data = data.sel(lat=lat[lat_cond])

    if region['lon_min'] < region['lon_max']:
        lon_cond = (lon >= region['lon_min']) & (lon <= region['lon_max'])
        return data.sel(lon=lon[lon_cond])
    else:
        # 处理跨越180°经线的情况
        cond1 = lon >= region['lon_min']
        cond2 = lon <= region['lon_max']
        part1 = data.sel(lon=lon[cond1])
        part2 = data.sel(lon=lon[cond2])
        return xr.concat([part1, part2], dim='lon')

# 循环处理每个区域
for name, reg in regions.items():
    print(f'处理区域: {name}')

    # 区域裁剪
    data_reg = subset_region(pre, lat, lon, reg)
    lat_reg = data_reg['lat']
    lon_reg = data_reg['lon']

    # 纬度加权
    coslat_reg = np.cos(np.deg2rad(lat_reg.values))
    wgts_reg = np.sqrt(coslat_reg)[:, np.newaxis]  # shape: (lat, 1)

    # 转为 numpy 数组
    data_array = data_reg.values  # shape: (time, lat, lon)
    data_array = (data_array - np.mean(data_array, axis=0)) / np.std(data_array, axis=0)
    # 检查 NaN
    if np.isnan(data_array).any():
        print(f"{name} 区域包含 NaN，请先进行缺失值处理。跳过此区域。\n")
        continue

    # EOF 分解
    solver = Eof(data_array, weights=wgts_reg)
    eof = solver.eofsAsCorrelation(neofs=10)
    pc = solver.pcs(npcs=10, pcscaling=1)
    var = solver.varianceFraction()

    # 保存 EOF 到 NetCDF
    ds = xr.Dataset({
        'eof': (('mode', 'lat', 'lon'), eof),
    }, coords={
        'lat': lat_reg,
        'lon': lon_reg,
        'mode': np.arange(1, eof.shape[0]+1)
    })
    ds.to_netcdf(f'{name}_eof.nc')
    print(f'{name}_eof.nc 已保存')

    # 保存 PC 和 方差贡献率为 CSV
    df_pc = pd.DataFrame(pc, columns=[f'PC{i+1}' for i in range(pc.shape[1])])
    df_var = pd.DataFrame({'variance_fraction': var})
    df_pc.to_csv(f'{name}_pc.csv', index=False)
    df_var.to_csv(f'{name}_variance.csv', index=False)
    print(f'{name}_pc.csv 和 {name}_variance.csv 已保存\n')
