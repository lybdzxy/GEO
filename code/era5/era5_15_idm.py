import csv
import os
from subprocess import call


def idmDownloader(task_url, folder_path, file_name):
    """
    IDM下载器
    :param task_url: 下载任务地址
    :param folder_path: 存放文件夹
    :param file_name: 文件名
    :return:
    """
    # IDM安装目录
    idm_engine = r"C:\Program Files (x86)\Internet Download Manager\IDMan.exe"
    # 将任务添加至队列
    call([idm_engine, '/d', task_url, '/p', folder_path, '/f', file_name, '/a'])
    # 开始任务队列
    call([idm_engine, '/s'])


def download_files_after_1950(csv_file_path, download_path):
    """
    读取CSV文件，筛选1950年后的数据，并通过IDM下载文件。

    参数:
    csv_file_path (str): CSV文件路径
    download_path (str): 下载文件保存路径
    """
    # 确保下载目录存在
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    with open(csv_file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                year = int(row['Year'].strip())
                month = int(row['Month'].strip())
                hour = int(row['Time'].split(':')[0].strip())  # 提取小时
                # 筛选1950年后的数据
                if year > 1900:
                    url = row['Download URL']
                    # 将Time转换为0z或6z格式
                    # time_str = row['Time'].split(':')[0] + 'z'
                    # 生成YearMonth格式，如195001
                    # year_month = f"{row['Year']}{int(row['Month']):02d}"
                    # 生成文件名，如ERA5_part_0z_lvl_195001.nc
                    filename = f"ERA5_part2_{hour}z_lvl_{year}{month:02d}.nc"
                    # 调用idmDownloader函数
                    try:
                        idmDownloader(url, download_path, filename)
                        print(f"Added to IDM: {filename}")
                    except Exception as e:
                        print(f"Failed to add {filename} to IDM: {e}")
            except ValueError:
                print(f"Invalid year format in row: {row}")
            except KeyError as e:
                print(f"Missing column in row: {row}, error: {e}")


# 示例用法
csv_file_path = 'download_missing_urls_2.csv'  # 替换为你的CSV文件路径
download_path = 'F:/ERA5/hourly/lvl/50s'  # 替换为你的下载目录

# download_path = 'F:/ERA5/hourly/lvl/50s'  # 替换为你的下载目录
download_files_after_1950(csv_file_path, download_path)