import xarray as xr
import geopandas as gpd
import salem

shp_dir = f'E:/GEO/geodata/CNTOT.shp'
ming_shp2 = gpd.read_file(shp_dir)

ssp_val=('126','245')
indicies = ('prcptot','r95p','r99p','r95ptot','r99ptot','sdii','rx1day','rx5day','r10mm','r20mm','cwd')
#for ssp in ssp_val:
for year in range(2015, 2023):
    for index in indicies:
        # 输入文件路径
        nc_path = f'E:/GEO/etccdi/his/{index}/{index}{year}.nc'
        output_nc_path = f'E:/GEO/etccdi/his/{index}/cut{index}{year}.nc'
        ds = xr.open_dataset(nc_path)
        ming = ds.salem.roi(shape=ming_shp2)
        # 获取要计算平均值的变量
        variable_names = ming.variables.keys()

        # 检查 '__xarray_dataarray_variable__' 是否在列表中
        if '__xarray_dataarray_variable__' in variable_names:
            variable_name = '__xarray_dataarray_variable__'
        elif 'tp' in variable_names:
            variable_name = 'tp'

        ming['tp'] = ming[variable_name].astype('float64')  # Ensure 'pr' is stored as double
        ming.to_netcdf(output_nc_path)


