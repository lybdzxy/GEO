import cdsapi
import time
import csv
import os
import logging
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.header import Header


# === 日志设置 ===
log_path = 'download_missing_log.txt'
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


cdsapirc_path = '.cdsapirc'
csv_path = 'download_missing_urls.csv'  # 保存下载链接的CSV路径
df_missing = pd.read_csv('missing_w_lvl.csv')

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
hour_to_time = {'00': '00:00', '06': '06:00', '12': '12:00', '18': '18:00'}

keys = [
    '12821cad-4f79-4c6f-bf8e-566b72ae96d7',
    'dc2ea968-45e0-46cd-91b3-9c6b34c0608c',
    'de7b1d61-02ed-41e4-b3e9-2960ea8de648',
    'f642eac2-6c56-49e1-8af5-5831b4d12b90'
]
key_index = 0

def send_email(subject, content, sender, receiver, smtp_server, smtp_port, password):
    # 创建邮件内容
    message = MIMEText(content, 'plain', 'utf-8')
    message['From'] = Header(sender)
    message['To'] = Header(receiver)
    message['Subject'] = Header(subject, 'utf-8')

    try:
        # 连接服务器并发送邮件
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.login(sender, password)
        server.sendmail(sender, [receiver], message.as_string())
        server.quit()
        print("✅ 邮件发送成功！")
    except Exception as e:
        print("❌ 邮件发送失败：", e)

if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Year', 'Month', 'Time', 'Download URL'])

# 下载循环
for index, row in df_missing.iterrows():
    try:
        year = row['Year']
        month = row['Month']
        hour = row['Hour']
        time_point = hour_to_time[str(hour).zfill(2)]

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

# 发送邮件提醒
send_email(
    subject="Python任务已完成提醒",
    content="你好，Python脚本任务已经执行完毕！",
    sender="tan81144703030@163.com",
    receiver="tan81144703030@163.com",
    smtp_server="smtp.163.com",
    smtp_port=465,
    password="AMhMJYiyxMNDpujB"  # 注意不是QQ邮箱登录密码
)
