import h5py
import numpy as np
import os
import glob
from scipy.interpolate import griddata
from osgeo import gdal, osr
import math

# 1. 读取经纬度栅格文件并裁剪 ROI
lats_all = np.fromfile('EASE2_M09km.lats.3856x1624x1.double', dtype=np.float64).reshape((1624, 3856))
lons_all = np.fromfile('EASE2_M09km.lons.3856x1624x1.double', dtype=np.float64).reshape((1624, 3856))

sel_col_start = 2784
sel_row_start = 256
sel_col_end = 2828
sel_row_end = 287

num_row = sel_row_end - sel_row_start
num_col = sel_col_end - sel_col_start
mid_row = math.ceil(num_row / 2)
mid_col = math.ceil(num_col / 2)

lats = lats_all[sel_row_start:sel_row_end, sel_col_start:sel_col_end]
lons = lons_all[sel_row_start:sel_row_end, sel_col_start:sel_col_end]

xlons_per_cell = abs(lons[mid_row][mid_col] - lons[mid_row - 1][mid_col - 1])
ylats_per_cell = abs(lats[mid_row][mid_col] - lats[mid_row - 1][mid_col - 1])

# 创建规则网格（ROI 范围）
grid_lon, grid_lat = lons, lats  # ROI 范围即为目标网格

# 2. 设置输入输出路径
work_dir = r"E:\SMAP_L4"
os.chdir(work_dir)
flist = glob.glob("*.h5")
output_dir = "roi_tif_output"
os.makedirs(output_dir, exist_ok=True)

# 3. 遍历处理每个 HDF5 文件
for fname in flist:
    print(f"处理文件：{fname}")
    try:
        with h5py.File(fname, 'r') as f:
            group = f['Soil_Moisture_Retrieval_Data']
            sm = group['soil_moisture'][:]
            lat = group['latitude'][:]
            lon = group['longitude'][:]

            mask = (sm != -9999.0) & ~np.isnan(sm) & ~np.isnan(lat) & ~np.isnan(lon)
            sm = sm[mask]
            lat = lat[mask]
            lon = lon[mask]

            if len(sm) == 0:
                print("无有效数据，跳过此文件")
                continue

            # 插值到 ROI 网格上
            print("正在插值...")
            grid_sm = griddata(
                np.column_stack((lon, lat)),
                sm,
                (grid_lon, grid_lat),
                method='linear'
            )

            rows, cols = grid_sm.shape
            driver = gdal.GetDriverByName('GTiff')
            output_path = os.path.join(output_dir, os.path.splitext(fname)[0] + '.tif')
            dataset = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)

            geotransform = [lons[0][0], xlons_per_cell, 0, lats[0][0], 0, -ylats_per_cell]
            dataset.SetGeoTransform(geotransform)

            srs = osr.SpatialReference()
            srs.SetWellKnownGeogCS("WGS84")
            dataset.SetProjection(srs.ExportToWkt())

            dataset.GetRasterBand(1).WriteArray(grid_sm)
            dataset.FlushCache()
            del dataset
            print("输出完成：", output_path)

    except Exception as e:
        print("处理出错：", fname)
        print("错误信息：", e)
