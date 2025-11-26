# 1.预处理
## 1.1.读取
nc:`readnc.py`

shp:`shpreader.py`

## 1.2.裁剪
nc:`nccut.py`原版

nc:`clip.py`检查variable，转换数据类型

## 1.3.分割
年份:`splityear.py`

## 1.4.合并
time:`nc_timemerge.py`

## 1.5.格式转换
tiff2nc:`tiff2nc.py`

## 1.5.ERA5日期后推
提取第一天:`era5firstday.py`
合并第一天和前一年:`era5correct.py`
时间前移一天:`era5year.py`

## 1.6.重采样
nc: `upscale.py`
## 1.7.下载
era5:`era5IDM.py`

## 1.8.坐标命名
nc:`coor_nc.py`

# 2.降尺度
## 2.1.Change Factor Downscaling
### 2.1.1.加法
原始:`cfmdownscaling.py`
改进: `cfmds_cal.py`
ssp-his: `cfmmyma.py`
模型时空重采样: `cfmre.py`
模型平均: `cfmmean.py`
降尺度（add+obs)（日文件）: `cfmdown.py`
日文件合并年文件同时检查2月29日: `cfmdownyear.py`

### 2.1.2.乘法
## 2.2.Quantile Perturbation Downscaling

# 3.ETCCDI Index计算
his: `etccdihis.py`
ssp: `etccdissp.py`
dask: `dask-etccdi.py`

# 4.Mann-Kendal test & Theil-Sen slope
全国平均线: `mann-kendal.py`
全国平均线: `mkt-tss.py`
全国nc: `mkt-tss-cntot.py`
全国制图: `mkt-tss-map.py`
全国制图（显著性）: `mkt-tss-sig-map.py`

# 5.eof
日最大降水量分解: `rx1dayeof.py`

# 6.test
`test`
`test2`
`test3`
`eof_test1`
`reof_test1`