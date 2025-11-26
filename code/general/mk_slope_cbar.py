import xarray as xr
import numpy as np

indices = ['prcptot', 'r95p', 'r99p', 'r95ptot', 'r99ptot', 'sdii', 'rx1day', 'r10mm', 'r20mm']
ssps = ['126', '245', '370', '585']
for index in indices:
    mi=np.array([])
    ma=np.array([])
    for ssp in ssps:
        data = xr.open_dataset(f'E:/GEO/result/qpm/cut/{ssp}{index}_mktest.nc')
        slope = data['slope']
        min = np.min(slope)
        max = np.max(slope)
        mi=np.append(min,mi)
        ma=np.append(max,ma)
    mii = np.min(mi)
    mai = np.max(ma)
    print(index,mii,mai)