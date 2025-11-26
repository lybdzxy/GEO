import xarray as xr
import geopandas as gpd
import salem

cmip6_path = f'E:/GEO/CMIP6/his/full'
file_names = ['pr_day_CESM2-WACCM_historical_r1i1p1f1_gn_full-time.nc', 'pr_day_CMCC-CM2-SR5_historical_r1i1p1f1_gn_full-time.nc', 'pr_day_CMCC-ESM2_historical_r1i1p1f1_gn_full-time.nc', 'pr_day_EC-Earth3-Veg-LR_historical_r1i1p1f1_gr_full-time.nc', 'pr_day_GFDL-ESM4_historical_r1i1p1f1_gr1_full-time.nc', 'pr_day_INM-CM4-8_historical_r1i1p1f1_gr1_full-time.nc', 'pr_day_INM-CM5-0_historical_r1i1p1f1_gr1_full-time.nc', 'pr_day_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_full-time.nc', 'pr_day_MRI-ESM2-0_historical_r1i1p1f1_gn_full-time.nc']

for cmip6 in file_names:
    # 输入文件路径
    nc_path = f'E:/GEO/CMIP6/his/full/{cmip6}'
    output_nc_path = f'E:/GEO/CMIP6/his/full/cut_{cmip6}'
    ds = xr.open_dataset(nc_path)
    print(ds)
    variable_names = ds.variables.keys()
    print(variable_names)
    pr = ds.sel(lat=slice(55, 15), lon=slice(70, 140))
    pr.to_netcdf(output_nc_path)

