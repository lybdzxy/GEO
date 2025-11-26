import xarray as xr
from xinvert import invert_Poisson
import numpy as np
import os

hour = [0, 6, 12, 18]

for year in range(2023, 2024):
    for month in range(6, 7):
        # 输出路径：一个月一个文件
        out_path = f"F:/ERA5/hourly/lvl/stream/ERA5_stream_1000hpa_{year}{month:02d}_test.nc"
        if os.path.exists(out_path):
            print(f"已存在: {out_path}，跳过")
            continue

        all_sf = []   # 保存 DataArray
        for h in hour:
            data_path = f'F:/ERA5/hourly/lvl/{h}z/ERA5_{h}z_lvl_{year}{month:02d}.nc'
            if not os.path.exists(data_path):
                print(f"缺少文件: {data_path}")
                continue

            # 打开文件并确保纬度递增
            ds = xr.open_dataset(data_path).sortby('latitude')
            lat = ds['latitude']
            lon = ds['longitude']
            time_dim = 'time' if 'time' in ds.dims else 'valid_time'
            level_dim = 'level' if 'level' in ds.dims else 'pressure_level'

            for t in ds[time_dim].values:
                vo = ds['vo'].sel({time_dim: t, level_dim: 1000})

                # xinvert 参数
                iParams = {
                    'BCs': ['extend', 'periodic'],
                    'mxLoop': 5000,
                    'tolerance': 1e-6,
                }

                # 计算流函数 (DataArray)
                sf = invert_Poisson(vo, dims=['latitude', 'longitude'], iParams=iParams)
                # 添加时间坐标
                sf = sf.expand_dims({time_dim: [t]})

                all_sf.append(sf)

        # 如果该月有数据，拼接 DataArray
        if all_sf:
            sf_data = xr.concat(all_sf, dim=time_dim)

            # ✅ 按时间排序
            sf_data = sf_data.sortby(time_dim)

            sf_data.name = 'streamfunction'
            sf_data.attrs = {
                'units': 'm^2 s^-1',
                'long_name': f'Streamfunction at 1000 hPa'
            }

            ds_out = xr.Dataset({'streamfunction': sf_data})
            ds_out.attrs = {
                'Conventions': 'CF-1.7',
                'source': f'Calculated from ERA5 vo at 1000 hPa using xinvert',
                'history': f'Created on {np.datetime_as_string(np.datetime64("now"), unit="s")}'
            }

            ds_out.to_netcdf(out_path)
            print(f"✅ 已保存: {out_path}")
        else:
            print(f"⚠️ {year}-{month:02d} 没有数据，跳过")
