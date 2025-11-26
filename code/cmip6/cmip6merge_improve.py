import os
import dask
import dask.bag as db
import xarray as xr
from dask.diagnostics import ProgressBar

def process_prefix(prefix, data_path, output_path, full_name):
    try:
        output_file = os.path.join(output_path, f'{prefix}.nc')
        if os.path.exists(output_file):
            print(f'{output_file} already exists, skipping.')
            return

        files = [os.path.join(data_path, name) for name in full_name if name.startswith(prefix)]
        datasets = [xr.open_dataset(file, chunks={'time': 10}) for file in files]
        combined = xr.concat(datasets, dim="time")
        combined.to_netcdf(output_file)
        print(f'{prefix} finish')
    except Exception as e:
        print(f'Error processing {prefix}: {e}')

if __name__ == '__main__':
    data_path = 'E:/GEO/CMIP6/tos/'
    output_path = 'E:/GEO/CMIP6/tos/fulltime/'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    full_name = [f for f in os.listdir(data_path) if f.endswith('.nc')]
    pre_name = list(set([f.split('_r1i1p1f1_')[0] for f in full_name]))

    # 使用dask.bag进行并行处理
    bag = db.from_sequence(pre_name)
    with ProgressBar():
        bag.map(process_prefix, data_path, output_path, full_name).compute()
