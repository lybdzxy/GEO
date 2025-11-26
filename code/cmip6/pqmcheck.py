import os

ssps = ['126', '245', '370', '585']
names = ['EC-Earth3', 'MPI-ESM1-2-HR', 'EC-Earth3-Veg',
         'CMCC-ESM2', 'GFDL-ESM4', 'INM-CM4-8',
         'BCC-CSM2-MR', 'MRI-ESM2-0', 'CMCC-CM2-SR5',
         'TaiESM1', 'NorESM2-MM', 'CESM2-WACCM', 'INM-CM5-0']

def check_output_files_exist(ssps, names, years):
    missing_files = []

    for ssp_num in ssps:
        for mod in names:
            for year in years:
                filename = f'E:/GEO/down/qpm/mid/qp_{mod}_{ssp_num}_{year}.nc'
                if not os.path.exists(filename):
                    missing_files.append(filename)

    if missing_files:
        print("以下输出文件缺失：")
        for missing_file in missing_files:
            print(missing_file)
    else:
        print("所有输出文件都存在。")

# 在调用时传入你的数据列表
check_output_files_exist(ssps, names, range(1961, 2015))
