import cdsapi
import time
import csv
import os
import logging
from wyx import send_email

# === 日志设置 ===
log_path = 'download_log_sfc_mon.txt'
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

cdsapirc_path = '.cdsapirc'
csv_path = 'download_sfc_mon.csv'

dataset = "reanalysis-era5-single-levels-monthly-means"
request = {
    "product_type": ["monthly_averaged_reanalysis"],
    'grid': '1.5/1.5',
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_dewpoint_temperature",
        "2m_temperature",
        "mean_sea_level_pressure",
        "mean_wave_period",
        "sea_surface_temperature",
        "surface_pressure",
        "total_precipitation",
        "mean_convective_precipitation_rate",
        "mean_evaporation_rate",
        "mean_large_scale_precipitation_rate",
        "mean_top_downward_short_wave_radiation_flux",
        "mean_top_net_long_wave_radiation_flux",
        "mean_top_net_long_wave_radiation_flux_clear_sky",
        "mean_top_net_short_wave_radiation_flux",
        "mean_top_net_short_wave_radiation_flux_clear_sky",
        "mean_total_precipitation_rate",
        "instantaneous_surface_sensible_heat_flux",
        "surface_latent_heat_flux",
        "surface_net_solar_radiation",
        "surface_net_solar_radiation_clear_sky",
        "surface_net_thermal_radiation",
        "surface_net_thermal_radiation_clear_sky",
        "surface_sensible_heat_flux",
        "surface_solar_radiation_downward_clear_sky",
        "surface_solar_radiation_downwards",
        "surface_thermal_radiation_downward_clear_sky",
        "surface_thermal_radiation_downwards",
        "toa_incident_solar_radiation",
        "top_net_solar_radiation",
        "top_net_solar_radiation_clear_sky",
        "top_net_thermal_radiation",
        "top_net_thermal_radiation_clear_sky",
        "total_sky_direct_solar_radiation_at_surface",
        "cloud_base_height",
        "high_cloud_cover",
        "low_cloud_cover",
        "medium_cloud_cover",
        "total_cloud_cover",
        "evaporation",
        "convective_precipitation",
        "convective_rain_rate",
        "large_scale_rain_rate",
        "large_scale_precipitation",
        "precipitation_type",
        "total_column_rain_water",
        "boundary_layer_height",
        "convective_available_potential_energy",
        "convective_inhibition",
        "geopotential",
        "total_column_water_vapour",
        "total_totals_index"
    ],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "time": ["00:00"],
    # "day": [f"{d:02d}" for d in range(1, 32)],
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
        writer.writerow(['Year', 'Download URL'])

with open(cdsapirc_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

with open(cdsapirc_path, 'w', encoding='utf-8') as file:
    for line in lines:
        if line.startswith('key:'):
            file.write(f'key: {keys[0]}\n')
        else:
            file.write(line)
client = cdsapi.Client()
send_email(content="你好，Python脚本任务开始执行！")

# 下载循环
for year in range(1940, 2025):
            try:
                client = cdsapi.Client()

                request['year'] = str(year)

                logging.info(f"开始请求：{year}")
                start = time.time()
                result = client.retrieve(dataset, request)
                download_url = result.location
                end = time.time()
                time_used = end - start

                with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([year, download_url])

                logging.info(f"下载完成：{download_url}（耗时 {time_used:.2f} 秒）")

                # 超时换 key
                if time_used > 300:
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

                    logging.info(f"超过300秒，切换 key -> 第 {key_index + 1} 个")

            except Exception as e:
                logging.error(f" 错误发生于 {year}：{str(e)}")

send_email(content="你好，Python脚本任务已经执行完毕！")
