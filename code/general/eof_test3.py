import numpy as np
import matplotlib.pyplot as plt
from eofs.standard import Eof

# 1. 生成原始数据
time = 10
lat = 4
lon = 5
data = np.random.rand(time, lat, lon)

# 2. 填充 NaN 值
data_with_nan = np.full((time, lat + 2, lon + 2), np.nan)
data_with_nan[:, 1:-1, 1:-1] = data

# 3. 执行 EOF 分析（含 NaN 数据）
solver_with_nan = Eof(data_with_nan)
eof_with_nan = solver_with_nan.eofsAsCorrelation(neofs=1)
pc_with_nan = solver_with_nan.pcs(npcs=1, pcscaling=1)

# 4. 填充 NaN 值并执行 EOF 分析
data_filled = np.nan_to_num(data_with_nan, nan=0)
solver_filled = Eof(data_filled)
eof_filled = solver_filled.eofsAsCorrelation(neofs=1)
pc_filled = solver_filled.pcs(npcs=1, pcscaling=1)

# 5. 可视化比较结果
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(eof_with_nan[0], cmap='coolwarm', interpolation='nearest')
plt.title('EOF with NaN')

plt.subplot(2, 2, 2)
plt.imshow(eof_filled[0], cmap='coolwarm', interpolation='nearest')
plt.title('EOF with Filled Data')

plt.subplot(2, 2, 3)
plt.plot(pc_with_nan)
plt.title('PC with NaN')

plt.subplot(2, 2, 4)
plt.plot(pc_filled)
plt.title('PC with Filled Data')

plt.tight_layout()
plt.show()
