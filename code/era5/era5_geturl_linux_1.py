import cdsapi
import time
import csv
import os
import logging

# === 日志设置 ===
log_path = 'download_log.txt'
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

cdsapirc_path = '.cdsapirc'
csv_path = 'download_urls.csv'

dataset = "reanalysis-era5-stream-levels"
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

keys = [
    '12821cad-4f79-4c6f-bf8e-566b72ae96d7',
    'dc2ea968-45e0-46cd-91b3-9c6b34c0608c',
    'de7b1d61-02ed-41e4-b3e9-2960ea8de648',
    'f642eac2-6c56-49e1-8af5-5831b4d12b90'
]
key_index = 0

if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Year', 'Month', 'Time', 'Download URL'])

# 下载循环
for year in range(1940, 2025):
    for month in range(1, 13):
        for time_point in ["00:00", "06:00", "12:00", "18:00"]:
            try:
                client = cdsapi.Client()

                request['year'] = str(year)
                request['month'] = f"{month:02d}"
                request['time'] = time_point

                logging.info(f"开始请求：{year}-{month:02d} {time_point}")
                start = time.time()
                result = client.retrieve(dataset, request)
                download_url = result.location
                end = time.time()
                time_used = end - start

                with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([year, month, time_point, download_url])

                logging.info(f"下载完成：{download_url}（耗时 {time_used:.2f} 秒）")

                # 超时换 key
                if time_used > 1800:
                    key_index = (key_index + 1) % len(keys)
                    next_key = keys[key_index]

                    with open(cdsapirc_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()

                    with open(cdsapirc_path, 'w', encoding='utf-8') as file:
                        for line in lines:
                            if line.startswith('key:'):
                                file.write(f'key: {next_key}\n')
                            else:
                                file.write(line)

                    logging.info(f"超过1800秒，切换 key -> 第 {key_index + 1} 个")

            except Exception as e:
                logging.error(f" 错误发生于 {year}-{month:02d} {time_point}：{str(e)}")
