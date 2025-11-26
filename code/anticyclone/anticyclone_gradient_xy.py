import torch
import numpy as np
import xarray as xr
import math
from sklearn.cluster import DBSCAN
from collections import defaultdict
import matplotlib.pyplot as plt

# 加载数据
file_path = 'E:/GEO/test/test_1.nc'  # 替换为你的nc文件路径
data = xr.open_dataset(file_path)

# 提取地表气压数据
pressure_data = data['sp'].squeeze().values  # 删除时间维度
latitudes = data['latitude'].values
longitudes = data['longitude'].values

# 常量
degree_to_km = 31  # 0.25° 大约等于 31 千米
threshold_gradient = 0.15  # 气压梯度的阈值 (Pa/km)
max_distance = 1200

# 将数据转换为张量，并移动到 GPU
pressure_tensor = torch.tensor(pressure_data, device='cuda')

# 为了处理边界点，使用填充
padded_tensor = torch.nn.functional.pad(pressure_tensor, (1, 1, 1, 1), mode='constant', value=-np.inf)

# 查找局部最大值：当前格点气压大于周围8个格点
local_maxima_mask = (
    (pressure_tensor > padded_tensor[:-2, :-2]) &
    (pressure_tensor > padded_tensor[1:-1, :-2]) &
    (pressure_tensor > padded_tensor[2:, :-2]) &
    (pressure_tensor > padded_tensor[:-2, 1:-1]) &
    (pressure_tensor > padded_tensor[2:, 1:-1]) &
    (pressure_tensor > padded_tensor[:-2, 2:]) &
    (pressure_tensor > padded_tensor[1:-1, 2:]) &
    (pressure_tensor > padded_tensor[2:, 2:]) &
    (padded_tensor[:-2, :-2] > 102000) &  # 上左
    (padded_tensor[1:-1, :-2] > 102000) &  # 上中
    (padded_tensor[2:, :-2] > 102000) &  # 上右
    (padded_tensor[:-2, 1:-1] > 102000) &  # 中左
    (padded_tensor[2:, 1:-1] > 102000) &  # 中右
    (padded_tensor[:-2, 2:] > 102000) &  # 下左
    (padded_tensor[1:-1, 2:] > 102000) &  # 下中
    (padded_tensor[2:, 2:] > 102000)  # 下右
)

# 计算气压梯度
dy, dx = torch.gradient(pressure_tensor)

# 将梯度从“每度”转换为“每千米”
dy = dy / degree_to_km
dx = dx / degree_to_km

# 为了处理边界点，使用填充对气压梯度进行填充
padded_dy = torch.nn.functional.pad(dy, (1, 1, 1, 1), mode='constant', value=-np.inf)
padded_dx = torch.nn.functional.pad(dx, (1, 1, 1, 1), mode='constant', value=-np.inf)

# 遍历局部最大值的掩码，判断其四个相邻点的气压梯度是否都大于阈值
valid_points_mask = local_maxima_mask.clone()  # 克隆局部最大值掩码

# 定义一个函数来计算方向上的气压梯度平均值
def calculate_average_gradient(gradient, index, direction):
    if direction == 'up':
        start_idx = max(0, index - 10)
        end_idx = index
    elif direction == 'down':
        start_idx = index + 1
        end_idx = min(gradient.shape[0], index + 11)
    elif direction == 'left':
        start_idx = max(0, index - 10)
        end_idx = index
    elif direction == 'right':
        start_idx = index + 1
        end_idx = min(gradient.shape[1], index + 11)

    return torch.mean(gradient[start_idx:end_idx])

# 使用填充的梯度来计算平均值并更新有效点掩码
for idx in range(10, pressure_tensor.shape[0] - 10):
    # 上（纬度方向）
    avg_grad_up = calculate_average_gradient(padded_dy, idx, 'up')
    # 下（纬度方向）
    avg_grad_down = calculate_average_gradient(padded_dy, idx, 'down')
    # 左（经度方向）
    avg_grad_left = calculate_average_gradient(padded_dx, idx, 'left')
    # 右（经度方向）
    avg_grad_right = calculate_average_gradient(padded_dx, idx, 'right')

    # 更新有效点掩码：判断四个方向的平均梯度绝对值是否大于阈值
    valid_points_mask[idx] &= (
        abs(avg_grad_up) > threshold_gradient and
        abs(avg_grad_down) > threshold_gradient and
        abs(avg_grad_left) > threshold_gradient and
        abs(avg_grad_right) > threshold_gradient
    )

# 获取有效点的坐标索引
valid_lat_indices, valid_lon_indices = torch.nonzero(valid_points_mask, as_tuple=True)

# 将索引转换为纬度和经度
valid_lats = latitudes[valid_lat_indices.cpu()]
valid_lons = longitudes[valid_lon_indices.cpu()]
valid_pressures = pressure_tensor[valid_lat_indices, valid_lon_indices].cpu()

# 输出有效点的坐标和气压值
valid_points = list(zip(valid_lats, valid_lons, valid_pressures.tolist()))

# Convert valid_points to a NumPy array
valid_points = np.array(valid_points)

# Filter out points with stream less than 101000
valid_points = valid_points[valid_points[:, 2] >= 102000]

valid_points_coor = valid_points[:, :2]

# 执行 DBSCAN 聚类
dbscan = DBSCAN(eps=9.67, min_samples=2)  # 您可能需要调整 eps 和 min_samples
cluster_labels = dbscan.fit_predict(valid_points_coor)

# 将聚类标签添加到 valid_points 数组中
valid_points_clustered = np.concatenate((valid_points, cluster_labels[:, np.newaxis]), axis=1)

# 创建一个字典来存储每个聚类中气压值最大的坐标和气压值
cluster_max_pressure = defaultdict(lambda: (-1, -1))  # 默认值为-1

for label in np.unique(cluster_labels):
    cluster_mask = valid_points_clustered[:, -1] == label
    cluster_data = valid_points_clustered[cluster_mask]

    max_pressure_idx = np.argmax(cluster_data[:, 2])  # 找到最大气压值的索引
    max_pressure_coord = cluster_data[max_pressure_idx, :2]  # 获取最大气压值的坐标
    max_pressure_value = cluster_data[max_pressure_idx, 2]  # 获取最大气压值

    # 存储每个聚类中气压值最大的坐标和气压值
    cluster_max_pressure[label] = (max_pressure_coord, max_pressure_value)

# 输出每个聚类中气压值最大的坐标和气压值
for label, (coord, pressure) in cluster_max_pressure.items():
    print(f"聚类 {label} 中气压值最大的坐标：{coord}, 气压值：{pressure}")

# 创建图形和子图
fig, ax = plt.subplots()

# 按照聚类标签循环绘制数据点
for label in np.unique(cluster_labels):
    cluster_mask = valid_points_clustered[:, -1] == label
    cluster_data = valid_points_clustered[cluster_mask]

    # 绘制当前聚类中的所有数据点
    ax.scatter(cluster_data[:, 1], cluster_data[:, 0], label=f'Cluster {label}', alpha=0.7)

    # 获取最大气压值点的坐标和气压值
    max_pressure_coord, max_pressure_value = cluster_max_pressure[label]

    # 根据最大气压值点的坐标添加标记
    ax.scatter(max_pressure_coord[1], max_pressure_coord[0], color='red', marker='x', label=f'Max Pressure: {int(max_pressure_value)}')

# 设置图例
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Clustered Data with Max Pressure Points')

# 显示图形
plt.show()
