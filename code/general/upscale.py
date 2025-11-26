import xarray as xr
from concurrent.futures import ProcessPoolExecutor

def resample_and_save(datapath, resultpath, resampling_factor):
    data = xr.open_dataset(datapath)
    resampled_data = data.coarsen(latitude=resampling_factor, longitude=resampling_factor, boundary='trim').mean()
    resampled_data.to_netcdf(resultpath)

ssp_values = [126, 245, 370, 585]
index_values = ['prcptot', 'r95p', 'r99p', 'r95ptot', 'r99ptot', 'sdii', 'rx1day', 'rx5day', 'r10mm', 'r20mm', 'cwd']

years = range(2015, 2080)
grid_values = [5, 10]

# Create a list of tasks for parallel processing
tasks = []
for ssp in ssp_values:
    for index in index_values:
        for year in years:
            for grid in grid_values:
                datapath = f'E:/GEO/etccdi/{ssp}/{index}/cut{index}{year}.nc'
                resultpath = f'E:/GEO/etccdi/{ssp}/{index}/cut{grid}x{grid}{index}{year}.nc'
                tasks.append((datapath, resultpath, grid))

# Set the number of parallel processes
num_processes = 4  # You can adjust this value based on your system's capabilities

if __name__ == '__main__':
    # Create a ProcessPoolExecutor and execute the tasks in parallel
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for task in tasks:
            executor.submit(resample_and_save, *task)