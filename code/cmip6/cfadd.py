import xarray as xr
import pandas as pd

pe = xr.open_dataset('E:/testidm/pr_day_NESM3_ssp126_r1i1p1f1_gn_20150101-20401231.nc')
peneed = pe.sel(time=slice("2020-1-1","2029-12-31"))
print(peneed)

monthly_avr = peneed.resample(time='1M').mean()

petest = pe.sel(time=slice("2030-1-1","2039-12-31"))
mon_test = petest.resample(time='1M').mean()
# 选择要修改的时间范围
start_date = pd.Timestamp('20200101')
end_date = pd.Timestamp('20291201')

# 创建新的时间数组
new_time = pd.date_range(start=start_date, end=end_date, freq='MS')

# 将时间数组转换为datetime64[ns]类型
new_time = pd.to_datetime(new_time)

# 使用新的时间数组替换数据集中的时间变量
mon_test = mon_test.assign_coords(time=new_time)
monthly_avr = monthly_avr.assign_coords(time=new_time)

print(mon_test)
print(monthly_avr)
cfadded=monthly_avr['pr'] - mon_test['pr']
print(cfadded)
mon_test.to_netcdf('mon_test.nc')
monthly_avr.to_netcdf('monthly_avr.nc')
cfadded.to_netcdf('output.nc')


