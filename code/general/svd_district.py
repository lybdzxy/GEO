from xMCA import xMCA
import xarray as xr
import pandas as pd
import geopandas as gpd
import salem

shp_dir = f'E:/GEO/result/pca/SE.shp'
se = gpd.read_file(shp_dir)

indices = ['prcptot', 'r95p', 'r99p', 'sdii', 'rx1day']
ssps = [126, 585]
frac_df_all = pd.DataFrame()

for ssp in ssps:
    for index in indices:
        # 定义文件路径
        tos_path = f'E:/GEO/CMIP6/tos/mme/tos_yearly_{ssp}.nc'
        prcptot_path = f'E:/GEO/etccdi/qpm/mme/full/{index}_{ssp}.nc'

        # 加载NetCDF文件
        tos = xr.open_dataset(tos_path)
        prcptot = xr.open_dataset(prcptot_path)
        prcptot_se = prcptot.salem.roi(shape=se)

        # 提取变量数据数组
        tos_array = tos['tos']
        prcptot_array = prcptot_se['pr']

        # 将时间坐标转换为字符串，然后重新分配时间坐标以确保一致
        tos_time_str = tos_array['time'].dt.strftime('%Y-%m-%d')
        prcptot_time_str = prcptot_array['time'].dt.strftime('%Y-%m-%d')

        tos_array['time'] = tos_time_str
        prcptot_array['time'] = prcptot_time_str

        # 确保时间坐标一致
        common_time = xr.merge([tos_array, prcptot_array])['time']

        tos_array = tos_array.sel(time=common_time)
        prcptot_array = prcptot_array.sel(time=common_time)
        tos_array = tos_array.sortby('time')
        prcptot_array = prcptot_array.sortby('time')
        # 使用xMCA库进行多变量奇异值分解分析
        tos_prcptot = xMCA(tos_array, prcptot_array)

        # 执行计算步骤，确保必要的属性被初始化
        tos_prcptot.solver()

        # 提取模式和扩展系数
        lp, rp = tos_prcptot.patterns(n=2)
        le, re = tos_prcptot.expansionCoefs(n=2)

        # 提取协方差分数
        frac = tos_prcptot.covFracs(n=2)

        # 定义保存路径
        lp_output_path = f'E:/GEO/result/qpm/svd/tos_{index}_{ssp}_se.nc'
        le_output_path = f'E:/GEO/result/qpm/svd/tos_{index}_{ssp}_se.xlsx'
        rp_output_path = f'E:/GEO/result/qpm/svd/{index}_tos_{ssp}_se.nc'
        re_output_path = f'E:/GEO/result/qpm/svd/{index}_tos_{ssp}_se.xlsx'

        # 将模式和扩展系数分别转换为 xarray 数据数组
        lp_da = xr.DataArray(lp, dims=['n', 'lon', 'lat'], name='leftPattern')
        rp_da = xr.DataArray(rp, dims=['n', 'lon', 'lat'], name='rightPattern')
        le_da = xr.DataArray(le, dims=['n', 'time'], name='leftExpansionCoefs')
        re_da = xr.DataArray(re, dims=['n', 'time'], name='rightExpansionCoefs')

        # 将左侧扩展系数(le)和右侧扩展系数(re)转换为 Pandas DataFrame
        le_df = pd.DataFrame({'time': le_da['time'].values, 'le_mode_0': le_da[0].values, 'le_mode_1': le_da[1].values})
        re_df = pd.DataFrame({'time': re_da['time'].values, 're_mode_0': re_da[0].values, 're_mode_1': re_da[1].values})

        # 根据经度重新排序lp和rp
        lp_sorted = lp_da.sortby('lon')
        rp_sorted = rp_da.sortby('lon')

        # 再根据纬度重新排序
        lp_sorted = lp_sorted.sortby('lat')
        rp_sorted = rp_sorted.sortby('lat')

        # 创建 xarray 数据集并保存为 NetCDF 文件
        lp_sorted.to_netcdf(lp_output_path)
        rp_sorted.to_netcdf(rp_output_path)
        le_df.to_excel(le_output_path, index=False)
        re_df.to_excel(re_output_path, index=False)

        # 将协方差分数转换为 Pandas DataFrame，并添加 SSP 和指数信息
        frac_df = pd.DataFrame({'Fraction_Mode_0': frac[0].values, 'Fraction_Mode_1': frac[1].values}, index=[0])
        frac_df['SSP'] = ssp
        frac_df['Index'] = index

        # 将当前循环中的协方差分数添加到整体 DataFrame 中
        frac_df_all = pd.concat([frac_df_all, frac_df], ignore_index=True)

# 定义保存路径
frac_output_path = 'E:/GEO/result/qpm/svd/all_frac_se.xlsx'

# 保存到 Excel 文件
frac_df_all.to_excel(frac_output_path, index=False)