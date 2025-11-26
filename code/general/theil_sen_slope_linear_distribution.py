import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

indices = ['prcptot', 'r95p', 'r99p', 'sdii', 'rx1day']

for index in indices:
    # 读取数据集
    ds_126_path = f'E:/GEO/result/ecm/126{index}_mktest.nc'
    ds_585_path = f'E:/GEO/result/ecm/585{index}_mktest.nc'

    ds_126 = xr.open_dataset(ds_126_path)
    ds_585 = xr.open_dataset(ds_585_path)

    # 提取 slope 变量
    ts_126 = ds_126['slope']
    ts_585 = ds_585['slope']

    # 将数据一维化并剔除 NaN
    ts_126_flaten = ts_126.values[~np.isnan(ts_126.values)]
    ts_585_flaten = ts_585.values[~np.isnan(ts_585.values)]

    # 计算百分之5和95百分位数
    p5_126, p95_126 = np.percentile(ts_126_flaten, [1, 99])
    p5_585, p95_585 = np.percentile(ts_585_flaten, [1, 99])

    # 选取百分之5到95百分位数的数据
    ts_126_filtered = ts_126_flaten[(ts_126_flaten >= p5_126) & (ts_126_flaten <= p95_126)]
    ts_585_filtered = ts_585_flaten[(ts_585_flaten >= p5_585) & (ts_585_flaten <= p95_585)]

    # 计算平均值、标准差
    mean_126 = np.mean(ts_126_filtered)
    mean_585 = np.mean(ts_585_filtered)
    std_126 = np.std(ts_126_filtered)
    std_585 = np.std(ts_585_filtered)
    # 分段并计数
    bins_126 = [
        np.sum((ts_126 < (0 - std_126)).values),
        np.sum(((ts_126 < 0) & (ts_126 > (0 - std_126))).values),
        np.sum(((ts_126 < (0 + 1.5 * std_126)) & (ts_126 > 0)).values),
        np.sum((ts_126 > (0 + 1.5 * std_126)).values)
    ]

    bins_585 = [
        np.sum((ts_585 < (0 - std_585)).values),
        np.sum(((ts_585 < 0) & (ts_585 > (0 - std_585))).values),
        np.sum(((ts_585 < (0 + 1.5 * std_585)) & (ts_585 > 0)).values),
        np.sum((ts_585 > (0 + 1.5 * std_585)).values)
    ]

    print(f'Index: {index}')
    print(
        f'126 Scenario: Less than -std: {bins_126[0]}, -std to 0: {bins_126[1]}, 0 to 1.5*std: {bins_126[2]}, Greater than 1.5*std: {bins_126[3]}')
    print(
        f'585 Scenario: Less than -std: {bins_585[0]}, -std to 0: {bins_585[1]}, 0 to 1.5*std: {bins_585[2]}, Greater than 1.5*std: {bins_585[3]}')
    # 合并两列数据
    ts_array = np.column_stack((ts_126_filtered, ts_585_filtered))

    # 绘制散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(ts_array[:, 0], ts_array[:, 1], alpha=0.5)

    # 绘制平均值点、n倍标准差
    for n in (-2,-1,0,1.5,3):
        plt.plot(+n*std_126, +n*std_585, 'bo', label=f'{n}*std')
    #plt.plot(mean_126, mean_585, 'ro', label='Mean value')
    plt.xlabel('Slope 126')
    plt.ylabel('Slope 585')
    plt.title(f'Scatter Plot of {index}')
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.close()
