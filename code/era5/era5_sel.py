import xarray as xr
import os
from tqdm import tqdm
import concurrent.futures

# 设置小时列表，表示0z, 6z, 12z, 18z的时段
hours = [0, 6, 12, 18]


# 定义处理单个文件的函数
def process_file(h, year, month):
    try:
        # 构建数据文件路径
        data_path = f'G:/ERA5/hourly/lvl/{h}z/ERA5_{h}z_lvl_{year}{month:02}.nc'

        # 构建输出文件路径
        output_path = f'E:/GEO/ERA5/lvl/ERA5_lvl_{year}{month:02}_{h:02}.nc'

        # 检查输出文件是否已存在，若存在则跳过该文件
        if os.path.exists(output_path):
            print(f"文件 {output_path} 已存在，跳过处理。")
            return

        # 确认输入文件是否存在
        if not os.path.exists(data_path):
            print(f"警告: 文件 {data_path} 不存在，跳过该文件。")
            return

        # 打开原始数据文件
        data = xr.open_dataset(data_path)

        # 选择850 hPa和500 hPa的level索引
        levels_of_interest = [850, 500, 200]
        level_index_850 = data.level.sel(level=850, method='nearest')
        level_index_500 = data.level.sel(level=500, method='nearest')
        level_index_200 = data.level.sel(level=200, method='nearest')

        # 提取850和500 hPa层的u, v, vo, z数据
        data_850_500_200 = data.sel(level=[level_index_850, level_index_500, level_index_200])

        # 只保留感兴趣的变量
        variables_to_keep = ['t', 'w', 'u', 'v', 'z', 'vo']
        data_sel = data_850_500_200[variables_to_keep]

        # 保存数据到新的NetCDF文件
        data_sel.to_netcdf(output_path)

        print(f"已处理并保存：{output_path}")

    except Exception as e:
        # 捕获异常并打印错误信息
        print(f"处理文件 {data_path} 时出错: {e}")


# 使用tqdm显示进度条并通过多进程进行并行处理
if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # 在进度条中遍历所有时段、年份和月份
        with tqdm(total=len(hours) * (2022 - 1980) * 12) as pbar:
            futures = []
            for h in hours:
                for year in range(1980, 2022):
                    for month in range(1, 13):
                        futures.append(executor.submit(process_file, h, year, month))

            # 显示进度
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)
