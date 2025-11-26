import cdsapi
import time
import csv
import os
import pandas as pd

cdsapirc_path = 'C:/Users/DELL/.cdsapirc'
csv_path = 'download_missing_urls.csv'  # 保存下载链接的CSV路径
df_missing = pd.read_csv('missing_files_1.csv')

dataset = "reanalysis-era5-stream-levels"
request = {
    "product_type": ["reanalysis"],
    'grid': '1.5/1.5',
    "variable": [
        "geopotential",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind"
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
hour_to_time = {'00': '00:00', '06': '06:00', '12': '12:00', '18': '18:00'}

# 四个 key 和轮换列表
keys = [
    '12821cad-4f79-4c6f-bf8e-566b72ae96d7',
    'dc2ea968-45e0-46cd-91b3-9c6b34c0608c',
    'de7b1d61-02ed-41e4-b3e9-2960ea8de648',
    'f642eac2-6c56-49e1-8af5-5831b4d12b90'
]
key_index = 2

# 如果 CSV 文件不存在，写入表头
if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Year', 'Month', 'Time', 'Download URL'])

# 获取数据
# 遍历缺失列表并发起下载
for index, row in df_missing.iterrows():
    year = row['Year']
    month = row['Month']
    hour = row['Hour']
    time_point = hour_to_time[str(hour).zfill(2)]

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
