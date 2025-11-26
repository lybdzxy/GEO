import cdsapi
import time
import csv
import os
'''
    "variable": [
        "geopotential",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
	"vertical_velocity"
    ],
'''
'''
    "variable": [
        'divergence',
        'potential_vorticity',
        'relative_humidity',
        'specific_humidity',
        'vorticity'
    ],
'''
cdsapirc_path = 'C:/Users/DELL/.cdsapirc'
csv_path = 'test.csv'  # 保存下载链接的CSV路径

dataset = "reanalysis-era5-pressure-levels"
request = {
    "product_type": ["reanalysis"],
    'grid': '1.5/1.5',
    "variable": [
        "geopotential",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity"
    ],
    "day": [f"{d:02d}" for d in range(1, 32)],
    "pressure_level": [
        "1", "2", "3", "5", "7", "10",
        "20", "30", "50", "70", "100", "125",
        "150", "175", "200", "225", "250", "300",
        "350", "400", "450", "500", "550", "600",
        "650", "700", "750", "775", "800", "825",
        "850", "875", "900", "925", "950", "975", "1000"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived"
}

# 四个 key 和轮换列表
keys = [
    '12821cad-4f79-4c6f-bf8e-566b72ae96d7',
    'dc2ea968-45e0-46cd-91b3-9c6b34c0608c',
    'de7b1d61-02ed-41e4-b3e9-2960ea8de648',
    'f642eac2-6c56-49e1-8af5-5831b4d12b90'
]
key_index = 0
key_start_no = 3
key_start = keys[key_start_no]

# 更新 .cdsapirc 文件
with open(cdsapirc_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

with open(cdsapirc_path, 'w', encoding='utf-8') as file:
    for line in lines:
        if line.startswith('key:'):
            file.write(f'key: {key_start}\n')
        else:
            file.write(line)

# 如果 CSV 文件不存在，写入表头
if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Year', 'Month', 'Time', 'Download URL'])
#F:/ERA5/hourly/lvl/18z/ERA5_18z_lvl_196005.nc

# 获取数据
for year in range(1960, 1961):
    for month in range(5, 6):
        for time_point in ["18:00"]:
            client = cdsapi.Client()

            request['year'] = str(year)
            request['month'] = f"{month:02d}"
            request['time'] = time_point

            start = time.time()
            result = client.retrieve(dataset, request)
            download_url = result.location
            end = time.time()
            time_used = end - start

            # ✅ 写入 CSV 文件
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([year, month, time_point, download_url])

            # ✅ 超时换 key（轮换）
            if time_used > 600:
                key_index = (key_index + 1) % len(keys)
                next_key = keys[key_index]

                # 更新 .cdsapirc 文件
                with open(cdsapirc_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()

                with open(cdsapirc_path, 'w', encoding='utf-8') as file:
                    for line in lines:
                        if line.startswith('key:'):
                            file.write(f'key: {next_key}\n')
                        else:
                            file.write(line)

                print(f"[INFO] Key 已切换为第 {key_index + 1} 个")
