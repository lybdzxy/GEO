import os
import cdsapi
import xarray as xr
from subprocess import call

# === 用户自定义：年份和月份范围 ===
YEAR_START = 1940  # 起始年份
YEAR_END = 1978    # 结束年份（包含）
MONTH_START = 1    # 起始月份
MONTH_END = 12     # 结束月份（包含）
OUTPUT_DIR = 'G:/ERA5/hourly/lvl/0z/'
USE_IDM = False    # 是否启用 IDM 下载（需在 Windows 上安装 IDM）

# 将压力层按批次分块，每次请求 n 层
LEVELS = [
    '1','2','3','5','7','10','20','30','50','70','100','125','150',
    '175','200','225','250','300','350','400','450','500','550','600',
    '650','700','750','775','800','825','850','875','900','925','950',
    '975','1000'
]
BATCH_SIZE = 10

def chunk_list(lst, n):
    """将列表分成每组 n 个元素的子列表"""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def idm_downloader(task_url, folder_path, file_name):
    """
    使用 IDM 下载器下载文件（仅 Windows 且已安装 IDM 时可用）
    :param task_url:  从 CDS API 获取的下载 URL
    :param folder_path: 本地存储目录
    :param file_name:    存储文件名
    :return:

    """
    idm_engine = r"C:\Program Files (x86)\Internet Download Manager\IDMan.exe"
    call([idm_engine, '/d', task_url, '/p', folder_path, '/f', file_name, '/a'])
    call([idm_engine, '/s'])


def merge_monthly(files, output_path):
    """使用 xarray 将按层分块下载的文件合并成一个月度文件"""
    ds_list = []
    for f in files:
        ds = xr.open_dataset(f)
        ds_list.append(ds)
    # 按 pressure_level 维度合并
    merged = xr.concat(ds_list, dim='level') if 'level' in ds_list[0].dims else xr.merge(ds_list)
    merged.to_netcdf(output_path)
    # 关闭并删除临时文件
    for ds in ds_list:
        ds.close()
    for f in files:
        os.remove(f)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    c = cdsapi.Client()

    base_request = {
        'product_type': ['reanalysis'],
        'data_format': 'netcdf',
        'grid': '1.5/1.5',
        'variable': [
            'divergence','geopotential','potential_vorticity','relative_humidity',
            'specific_humidity','temperature','u_component_of_wind',
            'v_component_of_wind','vertical_velocity','vorticity'
        ],
        'day': [f"{d:02d}" for d in range(1, 32)],
        'time': ['00:00'],
        'download_format': 'unarchived'
    }

    for year in range(YEAR_START, YEAR_END + 1):
        for month in range(MONTH_START, MONTH_END + 1):
            year_str = str(year)
            month_str = f"{month:02d}"
            monthly_output = os.path.join(OUTPUT_DIR, f"ERA5_0z_lvl_{year_str}{month_str}.nc")
            if os.path.exists(monthly_output):
                print(f"[跳过] 已存在月合并文件：{monthly_output}")
                continue

            # 按批次请求
            temp_files = []
            for idx, lev_batch in enumerate(chunk_list(LEVELS, BATCH_SIZE)):
                batch_str = '_'.join(lev_batch)
                tmp_fname = f"ERA5_0z_{year_str}{month_str}_lev{batch_str}.nc"
                tmp_path = os.path.join(OUTPUT_DIR, tmp_fname)
                if os.path.exists(tmp_path):
                    print(f"[跳过] 已存在临时文件：{tmp_path}")
                else:
                    req = base_request.copy()
                    req['year']=year_str
                    req['month']=month_str
                    req['pressure_level']=lev_batch

                    print(f"[下载] 批次 {idx+1}: 层 {batch_str} ...")
                    result = c.retrieve('reanalysis-era5-stream-levels', req, tmp_path)
                    if USE_IDM:
                        try:
                            url = result.location
                            idm_downloader(url, OUTPUT_DIR, tmp_fname)
                        except Exception:
                            print("[错误] IDM 下载失败，已通过 CDS API 保存本地")
                temp_files.append(tmp_path)

            # 合并当月所有临时文件
            print(f"[合并] {year_str}-{month_str} 所有层 ...")
            merge_monthly(temp_files, monthly_output)
            print(f"[完成] 月文件保存：{monthly_output}")

    print("所有任务完成。")

if __name__ == '__main__':
    main()
