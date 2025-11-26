import xarray as xr
import pandas as pd

indices = ['prcptot', 'r95p', 'r99p', 'r95ptot', 'r99ptot', 'sdii', 'rx1day', 'r10mm', 'r20mm']
w_path = 'E:/GEO/etccdi/qpm/CRI.xlsx'
ssps = ['126', '245', '370', '585']
models = ['BCC-CSM2-MR',
          'CMCC-CM2-SR5',
          'CMCC-ESM2',
          'EC-Earth3',
          'EC-Earth3-Veg',
          'GFDL-ESM4',
          'INM-CM4-8',
          'INM-CM5-0',
          'MPI-ESM1-2-HR',
          'MRI-ESM2-0',
          'NorESM2-MM',
          'TaiESM1'
          ]
for index in indices:
    df = pd.read_excel(w_path, sheet_name=index)
    print(models)
    for ssp in ssps:
        for year in range(2015, 2069):
            for mod in models:
                for i in range(0,13):
                    break