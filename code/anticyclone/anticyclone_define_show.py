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
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 20
# ============================
# 1. 数据读取与预处理
# ============================
file_path = 'E:/testidm/slp20216.nc'
data = xr.open_dataset(file_path)
data_sel = data.sel(valid_time='2021-06-01-12')

# 提取经度、纬度和海平面气压数据（转换为 hPa）
lon = data_sel['longitude'].values
lat = data_sel['latitude'].values
pressure = data_sel['msl'].values.squeeze() / 100  # 转换为 hPa
lat_filtered = lat[lat >= 0]
# 筛选出对应的气压数据
pressure_filtered = pressure[lat >= 0, :]
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

contours = plt.contour(lon, lat_filtered, pressure_above_1010, levels=np.arange(1010, np.nanmax(pressure), interval),
                       transform=ccrs.PlateCarree(), colors='black')

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

plt.close(fig)  # 关闭图像


# 筛选半径大于阈值的等压线
def calculate_radius(contour):
    center_x = np.mean(contour[:, 0])
    center_y = np.mean(contour[:, 1])
    distances = np.sqrt((contour[:, 0] - center_x) ** 2 + (contour[:, 1] - center_y) ** 2)
    return np.mean(distances)


min_radius_threshold = 0.5
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

# 辅助函数：将树转换为单行字符串表示（用于调试输出）
def tree_to_string(node):
    s = f"TreeNode({node.contour_index}, depth={node.depth}"
    if node.children:
        s += ", children=["
        s += ", ".join(tree_to_string(child) for child in node.children)
        s += "]"
    s += ")"
    return s

# ============================
# 6. 树清洗逻辑
# ============================
def clean_trees(trees):
    processed_nodes = set()
    filtered = [t for t in trees if t.depth >= 5]

    # 第二步：递归拆分树
    def split(tree):
        # 广度优先搜索寻找可拆分节点
        queue = [tree]
        while queue:
            current = queue.pop(0)
            if current.contour_index in processed_nodes:
                continue  # 跳过已处理的节点
            processed_nodes.add(current.contour_index)

            # 检查当前节点的子节点
            candidates = [c for c in current.children if c.depth > 3]
            if len(candidates) >= 2:
                return candidates  # 返回要拆分的子节点
            queue.extend(current.children)
        return None

    final_trees = []
    processing = filtered.copy()
    while processing:
        tree = processing.pop(0)
        to_split = split(tree)
        if to_split:
            processing.extend([c for c in to_split if c.depth >= 5])
        else:
            final_trees.append(tree)
            processed_nodes.add(tree_to_string(tree))

    # 第三步：修剪保留分支
    def prune(node):
        if not node.children:
            return node

        # 分离深度>3的子节点
        gt3 = [c for c in node.children if c.depth > 3]
        if len(gt3) >= 1:
            # 只保留深度最大的分支
            max_depth = max(c.depth for c in node.children)
            keep = [c for c in node.children if c.depth == max_depth]
        else:
            # 保留所有最大深度分支
            max_depth = max(c.depth for c in node.children)
            keep = [c for c in node.children if c.depth == max_depth]

        # 递归修剪
        new_node = TreeNode(node.contour_index)
        new_node.depth = node.depth
        for child in keep:
            new_child = prune(child)
            new_node.add_child(new_child)
        return new_node

    # 第四步：去重嵌套树
    unique_trees = []
    for tree in final_trees:
        is_nested = False
        for other_tree in final_trees:
            if other_tree != tree and is_subtree(tree, other_tree):
                is_nested = True
                break
        if not is_nested:
            unique_trees.append(tree)

    return [prune(t) for t in unique_trees]

# 辅助函数：判断是否是子树
def is_subtree(tree, parent_tree):
    if tree.contour_index == parent_tree.contour_index:
        return True
    for child in parent_tree.children:
        if is_subtree(tree, child):
            return True
    return False

cleaned_trees = clean_trees(trees)

def filter_small_outer_isobars(cleaned_trees, filtered_contours, min_radius=2.0):
    valid_trees = []
    for tree in cleaned_trees:
        outer_contour = filtered_contours[tree.contour_index]
        radius = calculate_radius(outer_contour)
        if radius >= min_radius:
            valid_trees.append(tree)
    return valid_trees


cleaned_trees = filter_small_outer_isobars(cleaned_trees, filtered_contours, min_radius=2.0)

# ============================
# 7. 提取高压中心和外层等压线
# ============================
def extract_outer_contours(cleaned_trees, filtered_contours):
    outer_list = []
    for idx, tree in enumerate(cleaned_trees):
        contour = filtered_contours[tree.contour_index]
        outer_list.append({
            'contour': contour,
            'tree_id': tree.contour_index,
            'label': f"H{idx + 1:02d}"
        })
    return outer_list


outer_contours = extract_outer_contours(cleaned_trees, filtered_contours)


def extract_inner_contours(cleaned_trees, filtered_contours):
    inner_contours = []
    for tree in cleaned_trees:
        stack = [tree]
        leaf_nodes = []
        while stack:
            node = stack.pop()
            if not node.children:
                leaf_nodes.append(node)
            else:
                stack.extend(node.children)
        for leaf in leaf_nodes:
            inner_contours.append({
                'contour': filtered_contours[leaf.contour_index],
                'tree_id': tree.contour_index
            })
    return inner_contours


inner_contours = extract_inner_contours(cleaned_trees, filtered_contours)


def calculate_high_centers(inner_contours, lon, lat, pressure):
    if lat[-1] < lat[0]:
        lat = lat[::-1]
        pressure = pressure[::-1, :]
    high_centers = {}
    for contour_data in inner_contours:
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
        if not np.isnan(grid_p) and grid_p >= 1010:
            if tree_id in high_centers:
                if grid_p > high_centers[tree_id]['stream']:
                    high_centers[tree_id] = {
                        'lon': grid_lon,
                        'lat': grid_lat,
                        'stream': grid_p,
                        'tree_id': tree_id
                    }
            else:
                high_centers[tree_id] = {
                    'lon': grid_lon,
                    'lat': grid_lat,
                    'stream': grid_p,
                    'tree_id': tree_id
                }
    return list(high_centers.values())


high_centers = calculate_high_centers(inner_contours, lon, lat_filtered, pressure_above_1010)
# ============================
# 8. 输出到CSV文件
# ============================
date_str = str(data_sel['valid_time'].values.astype('datetime64[h]')).replace('T', ' ')
outer_contour_dict = {oc['tree_id']: oc for oc in outer_contours}

csv_rows = []
for center in high_centers:
    tree_id = center['tree_id']
    outer_contour = outer_contour_dict.get(tree_id)
    if not outer_contour:
        continue

    contour_points = outer_contour['contour']
    points_str = ';'.join([f"{point[0]},{point[1]}" for point in contour_points])

    csv_rows.append({
        'date': date_str,
        'center_lon': f"{center['lon']:.2f}",
        'center_lat': f"{center['lat']:.2f}",
        'stream': f"{center['stream']:.2f}",
        'contour_points': points_str
    })

csv_path = 'high_pressure_systems_20200527_test3.csv'
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['date', 'center_lon', 'center_lat', 'stream', 'contour_points'])
    writer.writeheader()
    writer.writerows(csv_rows)

print(time.time()-start)

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# ============================
# 1. 绘制清洗前的等压线
# ============================
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=0)})
ax.set_extent(img_extent, crs=ccrs.PlateCarree())
ax.coastlines()

# 绘制未清洗的等压线
for contour in closed_contours:
    ax.plot(contour[:, 0], contour[:, 1], color='gray', linewidth=1, transform=ccrs.PlateCarree())

# ax.set_title('2021-06-01-12 清洗前的闭合等压线')

plt.show()

# ============================
# 绘制清洗后的树状结构包含的等压线
# ============================
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=0)})
ax.set_extent(img_extent, crs=ccrs.PlateCarree())
ax.coastlines()


def plot_tree_contours(tree, filtered_contours, ax, depth=1):
    # 绘制当前树的外层等压线
    outer_contour = filtered_contours[tree.contour_index]
    ax.plot(outer_contour[:, 0], outer_contour[:, 1], color=f'blue', linewidth=2, transform=ccrs.PlateCarree())

    # 对每个子节点递归绘制
    for child in tree.children:
        plot_tree_contours(child, filtered_contours, ax, depth + 1)


# 遍历所有树，绘制每棵树的等压线
for root in cleaned_trees:
    plot_tree_contours(root, filtered_contours, ax)

# ax.set_title('2021-06-01-12 清洗后的闭合等压线')

plt.show()

