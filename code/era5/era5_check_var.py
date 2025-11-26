import xarray as xr
import csv

list = []
for year in range(1940,2025):
    for month in range(1,13):
        for hour in (0,6,12,18):
            data_path = f'F:/ERA5/hourly/lvl/{hour}z/ERA5_{hour}z_lvl_{year}{month:02}.nc'
            data = xr.open_dataset(data_path)
            var = data.data_vars
            if 'w' in var:
                break
            else:
                list.append([year, month, hour])
output_csv = 'missing_w_lvl.csv'
# 将缺失的文件记录写入 CSV 文件
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Year', 'Month', 'Hour'])  # 表头
    writer.writerows(list)
