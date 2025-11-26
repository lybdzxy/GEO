import xarray as xr

for ssp in (126,585):
    data_path = f'E:/GEO/CMIP6/tos/mme/tos_Omon_ensemblemean_ssp{ssp}_r1i1p1f1_201501-210012_actual.nc'
    data = xr.open_dataset(data_path)
    print(data)
    tos = data['tos']
    # 进行时间切片，选取2015到2068年的数据
    tos = tos.sel(time=slice('2015-01-01', '2068-12-31'))
    # 计算年平均值
    pr_annual_mean = tos.resample(time='Y').mean()
    output_path = f'E:/GEO/CMIP6/tos/mme/tos_yearly_{ssp}.nc'
    pr_annual_mean.to_netcdf(output_path)