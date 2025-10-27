import csv
import xarray as xr
from matplotlib.path import Path
import warnings
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import concurrent.futures

warnings.filterwarnings("ignore")

# ============================
# 函数定义
# ============================

def calculate_radius(contour):
    """计算等压线的平均半径"""
    center_x = np.mean(contour[:, 0])
    center_y = np.mean(contour[:, 1])
    distances = np.sqrt((contour[:, 0] - center_x) ** 2 + (contour[:, 1] - center_y) ** 2)
    return np.mean(distances)

def build_containment_dict(filtered_contours):
    """构建等压线之间的包含关系字典"""
    containment_dict = {}
    paths = [Path(contour) for contour in filtered_contours]
    for i, outer_path in enumerate(paths):
        containment_dict[i] = []
        for j, inner_path in enumerate(paths):
            if i != j and np.all(outer_path.contains_points(inner_path.vertices)):
                containment_dict[i].append(j)
    return containment_dict

class TreeNode:
    """树节点类，用于表示等压线的层次结构"""
    def __init__(self, contour_index):
        self.contour_index = contour_index
        self.children = []
        self.depth = 1

    def add_child(self, child_node):
        self.children.append(child_node)

    def __repr__(self):
        return f"TreeNode({self.contour_index}, depth={self.depth}, children={len(self.children)})"

def build_trees(containment_dict):
    """根据包含关系构建树结构"""
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

def update_depth(node):
    """更新树的深度"""
    if node.children:
        child_depths = [update_depth(child) for child in node.children]
        node.depth = 1 + max(child_depths)
    return node.depth

def tree_to_string(node):
    """将树转换为字符串表示（用于去重）"""
    s = f"TreeNode({node.contour_index}, depth={node.depth}"
    if node.children:
        s += ", children=["
        s += ", ".join(tree_to_string(child) for child in node.children)
        s += "]"
    s += ")"
    return s

def clean_trees(trees):
    """清洗树结构：过滤、拆分、修剪、去重"""
    processed_nodes = set()
    filtered = [t for t in trees if t.depth >= 5]

    def split(tree):
        queue = [tree]
        while queue:
            current = queue.pop(0)
            if current.contour_index in processed_nodes:
                continue
            processed_nodes.add(current.contour_index)
            candidates = [c for c in current.children if c.depth > 3]
            if len(candidates) >= 2:
                return candidates
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

    def prune(node):
        if not node.children:
            return node
        gt3 = [c for c in node.children if c.depth > 3]
        if len(gt3) >= 1:
            max_depth = max(c.depth for c in node.children)
            keep = [c for c in node.children if c.depth == max_depth]
        else:
            max_depth = max(c.depth for c in node.children)
            keep = [c for c in node.children if c.depth == max_depth]
        new_node = TreeNode(node.contour_index)
        new_node.depth = node.depth
        for child in keep:
            new_child = prune(child)
            new_node.add_child(new_child)
        return new_node

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

def is_subtree(tree, parent_tree):
    """判断一棵树是否是另一棵树的子树"""
    if tree.contour_index == parent_tree.contour_index:
        return True
    for child in parent_tree.children:
        if is_subtree(tree, child):
            return True
    return False

def filter_small_outer_isobars(cleaned_trees, filtered_contours, min_radius=2.0):
    """过滤外层等压线半径小于阈值的树"""
    valid_trees = []
    for tree in cleaned_trees:
        outer_contour = filtered_contours[tree.contour_index]
        radius = calculate_radius(outer_contour)
        if radius >= min_radius:
            valid_trees.append(tree)
    return valid_trees

def extract_outer_contours(cleaned_trees, filtered_contours):
    """提取外层等压线"""
    outer_list = []
    for idx, tree in enumerate(cleaned_trees):
        contour = filtered_contours[tree.contour_index]
        outer_list.append({
            'contour': contour,
            'tree_id': tree.contour_index,
            'label': f"H{idx + 1:02d}"
        })
    return outer_list

def extract_inner_contours(cleaned_trees, filtered_contours):
    """提取内层等压线（高压中心对应的等压线）"""
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

def calculate_high_centers(inner_contours, lon, lat, pressure):
    """计算高压中心"""
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

# ============================
# 单日处理函数
# ============================

def process_single_day(target_time):
    """处理单日数据并返回高压系统识别结果"""
    try:
        # 1. 数据读取与预处理
        # 提取年份和月份
        year = target_time[:4]  # 例如 '2010'
        month = str(int(target_time[5:7]))  # 去掉前导零，例如 '01' -> '1'
        file_path = f'E:/testidm/slp{year}{month}.nc'  # 例如 'E:/testidm/slp20101.nc'
        data = xr.open_dataset(file_path)
        data_sel = data.sel(valid_time=target_time)

        lon = data_sel['longitude'].values
        lat = data_sel['latitude'].values
        pressure = data_sel['msl'].values.squeeze() / 100
        # 调整经度映射
        lon_adjusted = (lon + 180) % 360  # 使得原 180-360 变为 0-180，原 0-180 变为 180-360
        sort_idx = np.argsort(lon_adjusted)  # 获取新的索引顺序
        lon_adjusted = lon_adjusted[sort_idx]  # 重新排序

        # 重新调整 stream 数组的列顺序
        pressure = pressure[:, sort_idx]  # 按照调整后的经度顺序重新排列气压数据
        # 筛选纬度范围为 0° 到 90°N
        lat_filtered = lat[lat >= 0]

        # 筛选经度范围为 90° 到 270°E
        lon_mask = (lon_adjusted >= 90) & (lon_adjusted <= 270)
        lon_filtered = lon_adjusted[lon_mask]

        # 筛选出对应的气压数据
        pressure_filtered = pressure[lat >= 0, :][:, lon_mask]
        # 将气压低于 1010 hPa 的部分设为 nan
        pressure_above_1010 = np.where(pressure_filtered >= 1010, pressure_filtered, np.nan)

        # 2. 绘制等压线
        interval = 2
        img_extent = [90, 270, 20, 90]
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection=ccrs.NorthPolarStereo(central_longitude=0))
        ax.set_extent(img_extent, crs=ccrs.PlateCarree())
        contours = plt.contour(lon_filtered, lat_filtered, pressure_above_1010, levels=np.arange(1010, np.nanmax(pressure), interval),
                               transform=ccrs.PlateCarree(), colors='black')
        plt.close(fig)

        # 3. 查找闭合等压线
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

        # 4. 筛选半径大于阈值的等压线
        min_radius_threshold = 0.5
        filtered_contours = [contour for contour in closed_contours if calculate_radius(contour) >= min_radius_threshold]

        # 5. 构建包含关系字典
        containment_dict = build_containment_dict(filtered_contours)

        # 6. 构建树结构
        trees = build_trees(containment_dict)
        for root in trees:
            update_depth(root)

        # 7. 树清洗逻辑
        cleaned_trees = clean_trees(trees)
        cleaned_trees = filter_small_outer_isobars(cleaned_trees, filtered_contours, min_radius=2.0)

        # 8. 提取高压中心和外层等压线
        outer_contours = extract_outer_contours(cleaned_trees, filtered_contours)
        inner_contours = extract_inner_contours(cleaned_trees, filtered_contours)
        high_centers = calculate_high_centers(inner_contours, lon_filtered, lat_filtered, pressure_above_1010)
        # 4. 将高压中心和外层等压线的经度重新映射
        for center in high_centers:
            # 将高压中心的经度进行映射
            center['lon'] = (center['lon'] + 180) % 360

        for outer_contour in outer_contours:
            # 将外层等压线的经度进行映射
            contour_points_x = outer_contour['contour'][:, 0]
            contour_points_x_adjusted = (contour_points_x + 180) % 360
            outer_contour['contour'][:, 0] = contour_points_x_adjusted
        # 9. 准备CSV行数据
        date_str = target_time.replace('-', '').replace(' ', '').replace(':', '')[:12]
        outer_contour_dict = {oc['tree_id']: oc for oc in outer_contours}
        csv_rows = []
        for center in high_centers:
            tree_id = center['tree_id']
            outer_contour = outer_contour_dict.get(tree_id)
            if not outer_contour:
                continue
            contour_points = outer_contour['contour']
            points_str_x = ';'.join([f"{point[0]:.2f}" for point in contour_points])
            points_str_y = ';'.join([f"{point[1]:.2f}" for point in contour_points])
            csv_rows.append({
                'date': date_str,
                'center_lon': f"{center['lon']:.2f}",
                'center_lat': f"{center['lat']:.2f}",
                'stream': f"{center['stream']:.2f}",
                'contour_points_x': points_str_x,
                'contour_points_y': points_str_y
            })
        return csv_rows
    except Exception as e:
        print(f"错误处理日期 {target_time}: {str(e)}")
        return []

# ============================
# 主程序
# ============================

if __name__ == "__main__":
    # 生成2010-2025年的每日日期列表
    start_date = '1960-01-01'
    end_date = '2024-12-31'
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    target_times = [f"{date.strftime('%Y-%m-%d')}-{hour:02d}" for date in dates for hour in [0, 6, 12, 18]]
 
    # 输出文件路径
    output_file = 'high_pressure_systems_0360.csv'

    # 使用多进程并行处理
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['date', 'center_lon', 'center_lat', 'stream', 'contour_points_x', 'contour_points_y'])
        writer.writeheader()

        with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
            # 提交所有任务
            futures = {executor.submit(process_single_day, target_time): target_time for target_time in target_times}
            # 收集结果并写入CSV
            for future in concurrent.futures.as_completed(futures):
                target_time = futures[future]
                try:
                    csv_rows = future.result()
                    for row in csv_rows:
                        writer.writerow(row)
                except Exception as e:
                    print(f"错误处理日期 {target_time}: {str(e)}")
