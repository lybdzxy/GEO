import os
import xarray as xr
import pandas as pd

indices = ['prcptot', 'r95p', 'r99p', 'sdii', 'rx1day']
ssps = [126, 585]

for index in indices:
    for ssp in ssps:
        merged = None  # 将 merged 初始化放在更小的范围内
        for year in range(2015, 2069):
            file_path = f'E:/GEO/etccdi/qpm/mme/ecm/{index}_{ssp}_{year}.nc'
            if os.path.exists(file_path):
                data = xr.open_dataset(file_path)

                # 生成新的时间索引，使用该年最后一天的 0 时 0 分
                new_time = pd.date_range(f'{year}-12-31', periods=1, freq='D')
                data = data.assign_coords(time=new_time)

                if merged is None:
                    merged = data  # 如果 merged 为 None，直接将当前年份的数据赋值给 merged
                else:
                    merged = xr.concat([merged, data], dim="time")  # 否则，使用 xr.concat 连接数据
            else:
                print(f"文件 {file_path} 不存在，跳过该文件。")

        if merged is not None:
            # 如果需要重命名维度或变量名称，可以根据实际情况调整
            # merged = merged.rename({'latitude': 'lat', 'longitude': 'lon'})
            if '__xarray_dataarray_variable__' in merged.data_vars:
                merged = merged.rename({'__xarray_dataarray_variable__': 'pr'})

            output_path = f'E:/GEO/etccdi/qpm/mme/ecm/full/{index}_{ssp}.nc'
            merged.to_netcdf(output_path)
            print(f"保存 {index} {ssp} 数据到 {output_path}")
        else:
            print(f"没有找到任何 {index} {ssp} 数据，跳过保存。")
