import csv
import time
import xarray as xr
from matplotlib.path import Path
import warnings
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

start = time.time()
warnings.filterwarnings("ignore")

# ============================
# 1. 数据读取与预处理
# ============================
file_path = 'F:/ERA5/hourly/lvl/stream/ERA5_stream_1000hpa_202001.nc'
data = xr.open_dataset(file_path)
print(data)
data_sel = data.sel(time='2020-01-01-00')

# 提取经度、纬度和海平面气压数据（转换为 hPa）
lon = data_sel['longitude'].values
lat = data_sel['latitude'].values
stream = data_sel['streamfunction'].values.squeeze() / 10000
# ============================
# 2. 根据最大值条件进行筛选
# ============================
stream_max = np.nanmax(stream)  # 提取最大值
stream_min = np.nanmin(stream)
iqr = stream_max - stream_min  # 极差

if stream_max > 1000:
    # 若最大值大于 1000，则将流函数低于 0 的部分设为 nan
    stream_masked = np.where(stream >= 0, stream, np.nan)
else:
    # 若最大值小于等于 1000，则将低于 (最大值 - 四分之一极差) 的部分设为 nan
    threshold = stream_max - 0.25 * iqr
    stream_masked = np.where(stream >= threshold, stream, np.nan)

# ============================
# 2. 绘制等压线（用于获取轮廓数据）
# ============================
interval = (np.nanmax(stream)-np.nanmin(stream_masked)) / 20
img_extent = [-180, 180, 20, 90]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection=ccrs.NorthPolarStereo(central_longitude=0))
ax.set_extent(img_extent, crs=ccrs.PlateCarree())

contours = plt.contour(
    lon, lat, stream_masked,
    levels=np.arange(np.nanmin(stream_masked), np.nanmax(stream), interval),
    transform=ccrs.PlateCarree(), colors='black'
)

# ============================
# 3. 查找闭合等压线
# ============================
closed_contours = []
tolerance = 0.001
for c in contours.collections:
    for path in c.get_paths():
        vertices = path.vertices
        codes = path.codes
        if codes is None:
            if np.allclose(vertices[0], vertices[-1], atol=tolerance):
                closed_contours.append(vertices)
            continue
        sub_path_vertices = []
        for i, code in enumerate(codes):
            if code == Path.MOVETO:
                if sub_path_vertices and np.allclose(sub_path_vertices[0], sub_path_vertices[-1], atol=tolerance):
                    closed_contours.append(np.array(sub_path_vertices))
                sub_path_vertices = [vertices[i]]
            elif code == Path.LINETO:
                sub_path_vertices.append(vertices[i])
            elif code == Path.CLOSEPOLY:
                sub_path_vertices.append(vertices[i])
                if np.allclose(sub_path_vertices[0], sub_path_vertices[-1], atol=tolerance):
                    closed_contours.append(np.array(sub_path_vertices))
                sub_path_vertices = []
        if sub_path_vertices and np.allclose(sub_path_vertices[0], sub_path_vertices[-1], atol=tolerance):
            closed_contours.append(np.array(sub_path_vertices))

# plt.close(fig)  # 关闭图像
plt.show()


# 筛选半径大于阈值的等压线
def calculate_radius(contour):
    center_x = np.mean(contour[:, 0])
    center_y = np.mean(contour[:, 1])
    distances = np.sqrt((contour[:, 0] - center_x) ** 2 + (contour[:, 1] - center_y) ** 2)
    return np.mean(distances)


min_radius_threshold = 0.2
filtered_contours = [contour for contour in closed_contours if calculate_radius(contour) >= min_radius_threshold]


# ============================
# 4. 构建包含关系字典
# ============================
def build_containment_dict(filtered_contours):
    containment_dict = {}
    paths = [Path(contour) for contour in filtered_contours]
    for i, outer_path in enumerate(paths):
        containment_dict[i] = []
        for j, inner_path in enumerate(paths):
            if i != j:
                if np.all(outer_path.contains_points(inner_path.vertices)):
                    containment_dict[i].append(j)
    return containment_dict


containment_dict = build_containment_dict(filtered_contours)


# ============================
# 5. 构建树结构
# ============================
class TreeNode:
    def __init__(self, contour_index):
        self.contour_index = contour_index
        self.children = []
        self.depth = 1  # 初始深度为1

    def add_child(self, child_node):
        self.children.append(child_node)

    def __repr__(self):
        return f"TreeNode({self.contour_index}, depth={self.depth}, children={len(self.children)})"


def build_trees(containment_dict):
    nodes = {i: TreeNode(i) for i in containment_dict}
    all_indices = set(containment_dict.keys())
    contained_indices = set()
    for contained in containment_dict.values():
        contained_indices.update(contained)
    root_indices = list(all_indices - contained_indices)

    for i, children in containment_dict.items():
        node = nodes[i]
        for child_index in children:
            node.add_child(nodes[child_index])
    return [nodes[r] for r in root_indices]


trees = build_trees(containment_dict)


def update_depth(node):
    if node.children:
        child_depths = [update_depth(child) for child in node.children]
        node.depth = 1 + max(child_depths)
    else:
        node.depth = 1
    return node.depth


for root in trees:
    update_depth(root)


# ============================
# 6. 新树清洗逻辑（简化版）
# ============================
def clean_trees(trees):
    # 只保留深度 >= 5 的树
    return [t for t in trees if t.depth >= 0]


cleaned_trees = clean_trees(trees)


# ============================
# 7. 提取潜在高压中心（所有叶节点）
# ============================
def extract_leaf_contours(cleaned_trees, filtered_contours):
    leaf_contours = []
    seen_contour_indices = set()  # 用于去重
    for tree in cleaned_trees:
        stack = [tree]
        while stack:
            node = stack.pop()
            if not node.children and node.contour_index not in seen_contour_indices:
                leaf_contours.append({
                    'contour': filtered_contours[node.contour_index],
                    'tree_id': tree.contour_index
                })
                seen_contour_indices.add(node.contour_index)
            else:
                stack.extend(node.children)
    return leaf_contours


leaf_contours = extract_leaf_contours(cleaned_trees, filtered_contours)


def calculate_potential_high_centers(leaf_contours, lon, lat, pressure):
    if lat[-1] < lat[0]:
        lat = lat[::-1]
        pressure = pressure[::-1, :]
    potential_centers = []
    seen_centers = set()  # 用于去重中心点
    for contour_data in leaf_contours:
        contour = contour_data['contour']
        tree_id = contour_data['tree_id']
        center_lon = np.mean(contour[:, 0])
        center_lat = np.mean(contour[:, 1])
        center_lon = np.clip(center_lon, lon.min(), lon.max())
        center_lat = np.clip(center_lat, lat.min(), lat.max())
        lon_idx = np.argmin(np.abs(lon - center_lon))
        lat_idx = np.argmin(np.abs(lat - center_lat))
        grid_lon = lon[lon_idx]
        grid_lat = lat[lat_idx]
        grid_p = pressure[lat_idx, lon_idx]
        if not np.isnan(grid_p):
            center_tuple = (grid_lon, grid_lat, grid_p)
            if center_tuple not in seen_centers:
                potential_centers.append({
                    'lon': grid_lon,
                    'lat': grid_lat,
                    'stream': grid_p,
                    'tree_id': tree_id
                })
                seen_centers.add(center_tuple)
    return potential_centers


potential_high_centers = calculate_potential_high_centers(leaf_contours, lon, lat, stream_masked)

# ============================
# 8. 输出到CSV文件（含聚类）
# ============================
date_str = str(data_sel['time'].values.astype('datetime64[h]')).replace('T', ' ')


# 自定义聚类：经纬度差≤5度视为邻居
def cluster_points(points, eps=5.0):
    n = len(points)
    visited = [False] * n
    clusters = []

    def dfs(i, cluster):
        visited[i] = True
        cluster.append(i)
        for j in range(n):
            if not visited[j]:
                lon_i, lat_i = points[i]['lon'], points[i]['lat']
                lon_j, lat_j = points[j]['lon'], points[j]['lat']
                if abs(lon_i - lon_j) <= eps and abs(lat_i - lat_j) <= eps:
                    dfs(j, cluster)

    for i in range(n):
        if not visited[i]:
            cluster = []
            dfs(i, cluster)
            clusters.append(cluster)

    # 每个簇保留stream最大的点
    final_points = []
    for cluster in clusters:
        if cluster:
            cluster_points = [points[i] for i in cluster]
            max_point = max(cluster_points, key=lambda p: p['stream'])
            final_points.append(max_point)

    return final_points


# 应用聚类
if potential_high_centers:
    final_centers = cluster_points(potential_high_centers, eps=5.0)
else:
    final_centers = potential_high_centers

# 生成CSV行
csv_rows = []
for center in final_centers:
    csv_rows.append({
        'date': date_str,
        'center_lon': f"{center['lon']:.2f}",
        'center_lat': f"{center['lat']:.2f}",
        'stream': f"{center['stream']:.2f}"
    })
print(csv_rows)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection=ccrs.NorthPolarStereo(central_longitude=0))
ax.set_extent(img_extent, crs=ccrs.PlateCarree())

# 绘制背景气压场（可选）
cf = ax.contourf(
    lon, lat, stream_masked,
    levels=np.arange(np.nanmin(stream_masked), np.nanmax(stream_masked), interval),
    transform=ccrs.PlateCarree(),
    cmap='Blues',
    alpha=0.5
)

# 绘制清洗后的等压线
for contour in filtered_contours:
    ax.plot(contour[:, 0], contour[:, 1], color='black', linewidth=1.5, transform=ccrs.PlateCarree())

# 绘制高压中心
for center in final_centers:
    ax.plot(center['lon'], center['lat'], 'ro', markersize=10, transform=ccrs.PlateCarree())
    ax.text(center['lon'], center['lat'], f"{center['stream']:.1f}", transform=ccrs.PlateCarree(),
            fontsize=12, color='red', verticalalignment='bottom', horizontalalignment='right')

# 添加地图特征
ax.gridlines()

plt.show()
'''csv_path = 'potential_high_pressure_centers_20200101.csv'
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['date', 'center_lon', 'center_lat', 'stream'])
    writer.writeheader()
    writer.writerows(csv_rows)
'''
print(time.time() - start)