import os
import xarray as xr

data_path = 'E:/GEO/CMIP6/tos/'
output_path = 'E:/GEO/CMIP6/tos/fulltime/'

if not os.path.exists(output_path):
    os.makedirs(output_path)

full_name = [f for f in os.listdir(data_path) if f.endswith('.nc')]
pre_name = list(set([f.split('_r1i1p1f1_')[0] for f in full_name]))

for prefix in pre_name:
    try:
        output_file = os.path.join(output_path, f'{prefix}.nc')
        if os.path.exists(output_file):
            print(f'{output_file} already exists, skipping.')
            continue

        merged = None
        files = [name for name in full_name if name.startswith(prefix)]
        for file in files:
            file_path = os.path.join(data_path, file)
            data = xr.open_dataset(file_path)
            if merged is None:
                merged = data
            else:
                merged = xr.concat([merged, data], dim="time")
        merged.to_netcdf(output_file)
        print(f'{prefix} finish')
    except Exception as e:
        print(f'Error processing {prefix}: {e}')
