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
#file_path = 'E:/testidm/slp20201.nc'
file_path = 'F:/ERA5/hourly/sfc/fin/ERA5_00z_sfc_202001_instant.nc'
data = xr.open_dataset(file_path)
data_sel = data.sel(valid_time='2020-01-07-00')

# 提取经度、纬度和海平面气压数据（转换为 hPa）
lon = data_sel['longitude'].values
lat = data_sel['latitude'].values
pressure = data_sel['msl'].values.squeeze() / 100  # 转换为 hPa
# 调整经度：将180-360映射到0-180，0-180映射到180-360
lon_adjusted = np.where(lon >= 180, lon - 180, lon + 180)
sort_idx = np.argsort(lon_adjusted)  # 按调整后的经度排序
lon_adjusted = lon_adjusted[sort_idx]
pressure = pressure[:, sort_idx]  # 按照调整后的经度顺序重新排列气压数据

# 筛选经度范围为 90° 到 270°E，以覆盖本初子午线两侧
lon_mask = (lon_adjusted >= 90) & (lon_adjusted <= 270)
lon_filtered = lon_adjusted[lon_mask]
pressure_filtered = pressure[:, lon_mask]

# 将气压低于 1010 hPa 的部分设为 nan
pressure_above_1010 = np.where(pressure_filtered >= 1010, pressure_filtered, np.nan)

# ============================
# 2. 绘制等压线（用于获取轮廓数据）
# ============================
interval = 2
img_extent = [-180, 180, 20, 90]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection=ccrs.NorthPolarStereo(central_longitude=0))
ax.set_extent(img_extent, crs=ccrs.PlateCarree())

contours = plt.contour(
    lon_filtered, lat, pressure_above_1010,
    levels=np.arange(1010, np.nanmax(pressure), interval),
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

# ============================
# 4. 筛选半径大于阈值的等压线
# ============================
def calculate_radius(contour):
    center_x = np.mean(contour[:, 0])
    center_y = np.mean(contour[:, 1])
    distances = np.sqrt((contour[:, 0] - center_x) ** 2 + (contour[:, 1] - center_y) ** 2)
    return np.mean(distances)

min_radius_threshold = 0.5
filtered_contours = [contour for contour in closed_contours if calculate_radius(contour) >= min_radius_threshold]

# ============================
# 5. 构建包含关系字典
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
# 6. 构建树结构
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
    return node.depth

for root in trees:
    update_depth(root)

# ============================
# 7. 新树清洗逻辑（简化版）
# ============================
def clean_trees(trees):
    # 只保留深度 >= 5 的树
    return [t for t in trees if t.depth >= 5]

cleaned_trees = clean_trees(trees)

# ============================
# 8. 提取潜在高压中心（所有叶节点）
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
        if not np.isnan(grid_p) and grid_p > 1020:
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

potential_high_centers = calculate_potential_high_centers(leaf_contours, lon_filtered, lat, pressure_above_1010)

# ============================
# 9. 输出到CSV文件
# ============================
date_str = str(data_sel['valid_time'].values.astype('datetime64[h]')).replace('T', ' ')

csv_rows = []
for center in potential_high_centers:
    csv_rows.append({
        'date': date_str,
        'center_lon': f"{center['lon']:.2f}",
        'center_lat': f"{center['lat']:.2f}",
        'stream': f"{center['stream']:.2f}"
    })

# 绘制背景气压场（可选）
cf = ax.contourf(
    lon_filtered, lat, pressure_above_1010,
    levels=np.arange(np.nanmin(pressure_above_1010), np.nanmax(pressure_above_1010), interval),
    transform=ccrs.PlateCarree(),
    cmap='Blues',
    alpha=0.5
)

# 绘制清洗后的等压线
for contour in filtered_contours:
    ax.plot(contour[:, 0], contour[:, 1], color='black', linewidth=1.5, transform=ccrs.PlateCarree())

# 绘制高压中心
for center in potential_high_centers:
    ax.plot(center['lon'], center['lat'], 'ro', markersize=10, transform=ccrs.PlateCarree())
    ax.text(center['lon'], center['lat'], f"{center['stream']:.1f}", transform=ccrs.PlateCarree(),
            fontsize=12, color='red', verticalalignment='bottom', horizontalalignment='right')

# 添加地图特征
ax.gridlines()

plt.show()
print(time.time() - start)