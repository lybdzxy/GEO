import cdsapi
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


if __name__ == '__main__':
    c = cdsapi.Client()  # 创建用户

    # 数据信息字典
    dic = {
    "product_type": ["reanalysis"],
    "variable": ["surface_pressure"],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "00:00", "03:00", "06:00",
        "09:00", "12:00", "15:00",
        "18:00", "21:00"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived"
}

    for y in range(2004,2014):  # 遍历年
        dic['year'] = str(y)

        r = c.retrieve("reanalysis-era5-single-levels", dic, )  # 文件下载器
        url = r.location  # 获取文件下载地址
        path = r'E:\testidm'  # 存放文件夹
        filename = 'sp' + str(y) + '.nc'  # 文件名
        idmDownloader(url, path, filename)  # 添加进IDM中下载