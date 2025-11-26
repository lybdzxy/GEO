import xarray as xr
import xclim.indices as indices
import numpy as np
ssp_val=(126,245)
for ssp in ssp_val:
    for year in range(2015, 2080):
        input_path = f'E:/GEO/downed/{ssp}/{year}.nc'
        # 打开包含 ERA5 数据的 NetCDF 文件
        ds = xr.open_dataset(input_path)

        # 选择特定的经度和纬度范围
        pr = ds['pr'].sel(latitude=slice(60, 15), longitude=slice(70, 140))

        # 提取降水数据，假设数据变量名为 'pr'（总降水量，单位为 mm/day）
        pr.attrs['units'] = 'mm/day'

        # etccdi计算
        # 计算 PRCPTOT（年降水总量）
        prcp = indices.prcptot(pr)
        prcp.to_netcdf(f'E:/GEO/etccdi/{ssp}/prcptot/prcptot{year}.nc')
        # 计算 R95p（95th percentile of daily precipitation）
        p95 = pr.quantile(0.95, dim="time", keep_attrs=True)
        r95p = indices._anuclim.prcptot(pr, p95)
        r95p.to_netcdf(f'E:/GEO/etccdi/{ssp}/r95p/r95p{year}.nc')
        # 计算 R99p（99th percentile of daily precipitation）
        p99 = pr.quantile(0.99, dim="time", keep_attrs=True)
        r99p = indices._anuclim.prcptot(pr, p99)
        r99p.to_netcdf(f'E:/GEO/etccdi/{ssp}/r99p/r99p{year}.nc')
        # 计算 r95ptot,r99ptot
        r95ptot = r95p / prcp
        r95ptot.to_netcdf(f'E:/GEO/etccdi/{ssp}/r95ptot/r95ptot{year}.nc')
        r99ptot = r99p / prcp
        r99ptot.to_netcdf(f'E:/GEO/etccdi/{ssp}/r99ptot/r99ptot{year}.nc')
        # 计算 SDII（平均降水日降水量）
        sdii = indices.daily_pr_intensity(pr)
        sdii.to_netcdf(f'E:/GEO/etccdi/{ssp}/sdii/sdii{year}.nc')
        # 计算 Rx1day（连续 1 天内的最大日降水量）
        rx1day = indices.max_n_day_precipitation_amount(pr, window=1)
        rx1day.to_netcdf(f'E:/GEO/etccdi/{ssp}/rx1day/rx1day{year}.nc')
        # 计算 Rx5day（连续 5 天内的最大日降水量）
        rx5day = indices.max_n_day_precipitation_amount(pr, window=5)
        rx5day.to_netcdf(f'E:/GEO/etccdi/{ssp}/rx5day/rx5day{year}.nc')
        # 计算 R10mm（连续 1 天中 10 mm 以上降水的次数）
        r10mm = ((pr > 10).groupby('time.dayofyear').sum(dim='time') > 0).sum(dim='dayofyear')
        # days_over_thresh = indices.days_over_precip_thresh(precip, thresh=20)
        r10mm.to_netcdf(f'E:/GEO/etccdi/{ssp}/r10mm/r10mm{year}.nc')
        # 计算 R20mm（连续 1 天中 20 mm 以上降水的次数）
        r20mm = ((pr > 20).groupby('time.dayofyear').sum(dim='time') > 0).sum(dim='dayofyear')
        # days_over_thresh = indices.days_over_precip_thresh(precip, thresh=20)
        r20mm.to_netcdf(f'E:/GEO/etccdi/{ssp}/r20mm/r20mm{year}.nc')
        # 计算 CWD （连续最大降水大于1mm的天数）
        cwd = indices.wet_spell_max_length(pr)
        cwd.to_netcdf(f'E:/GEO/etccdi/{ssp}/cwd/cwd{year}.nc')
