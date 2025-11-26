import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

# 设置变量索引和 SSPs
indices = ['prcptot', 'r95p', 'r99p', 'sdii', 'rx1day']
ssps = [126, 585]

# 百分位数
percentiles = [5, 95]

# 创建一个新的图形对象和子图
fig, axes = plt.subplots(len(ssps), len(indices), figsize=(15, 10))

# 遍历所有索引和 SSPs
for i, ssp in enumerate(ssps):
    for j, index in enumerate(indices):
        rp_output_path = f'E:/GEO/result/ecm/{index}_tos_{ssp}.nc'

        # 读取 NetCDF 文件
        dataset = nc.Dataset(rp_output_path)

        # 获取变量名（假设每个文件都有名为 'rightPattern' 的变量）
        var_name = 'rightPattern'
        var_data = dataset.variables[var_name][:]

        # 将数据转换为一维数组
        data_flat = var_data.compressed()

        # 绘制箱线图
        bp = axes[i, j].boxplot(data_flat, patch_artist=True)

        # 获取箱线图的 Y 范围
        ymin, ymax = axes[i, j].get_ylim()

        # 添加水平线以标注特定的百分位数
        for p in percentiles:
            value = np.percentile(data_flat, p, interpolation='nearest')
            axes[i, j].axhline(value, color='gray', linestyle='--', linewidth=0.5)
            axes[i, j].text(1.05, value, f'{p}%', color='gray', fontsize=8, transform=axes[i, j].get_yaxis_transform())

        axes[i, j].set_title(f'{index}, SSP {ssp}', fontsize=10)

        # 关闭文件
        dataset.close()

# 调整布局以防止标签重叠
plt.tight_layout()
plt.show()
