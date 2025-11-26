import pandas as pd
import xarray as xr
import salem
import geopandas as gpd

# 读取 Excel 文件中的数据存储到字典中
w_path = 'E:/GEO/etccdi/qpm/mul_rwb.xlsx'
indices = ['prcptot', 'r95p', 'r99p', 'sdii', 'rx1day']
index_data = {}
wi = {}
shp_dir = 'E:/GEO/result/geodata/ecm_el.shp'
shp = gpd.read_file(shp_dir)

for index in indices:
    df = pd.read_excel(w_path, sheet_name=index)
    index_data[index] = df
    wi[index] = index_data[index]['Wi'].values

models = ['BCC-CSM2-MR','CMCC-CM2-SR5','CMCC-ESM2',
            'EC-Earth3','EC-Earth3-Veg','GFDL-ESM4',
            'INM-CM4-8','INM-CM5-0','MPI-ESM1-2-HR',
            'MRI-ESM2-0','NorESM2-MM','TaiESM1']
ssps = ['126', '585']

for index in indices:
    for ssp in ssps:
        for year in range(2015, 2069):
            mme = None  # Initialize mme
            for mod in models:
                data_path = f'E:/GEO/etccdi/qpm/{index}_{mod}_{ssp}_{year}.nc'
                data = xr.open_dataset(data_path)
                if 'time' in data.dims:
                    data = data.transpose('time', 'lat', 'lon')
                else:
                    data = data.transpose('lat', 'lon')
                data = data.salem.roi(shape=shp)
                wighted_data = data['__xarray_dataarray_variable__'] * wi[index][models.index(mod)]  # Corrected indexing
                if mme is None:
                    mme = wighted_data.copy()  # Make sure to copy the data to avoid reference issues
                else:
                    mme += wighted_data  # Accumulate the weighted data
            output_path = f'E:/GEO/etccdi/qpm/mme/ecm/{index}_{ssp}_{year}.nc'
            mme.to_netcdf(output_path)  # Corrected method name
