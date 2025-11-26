import csv
import time
import xarray as xr
from matplotlib.path import Path
import warnings
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import concurrent.futures
from tqdm import tqdm

start = time.time()
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

def clean_trees(trees):
    """清洗树结构：只保留深度 >= 5 的树"""
    return [t for t in trees if t.depth >= 5]

def extract_leaf_contours(cleaned_trees, filtered_contours):
    """提取潜在高压中心（所有叶节点）"""
    leaf_contours = []
    seen_contour_indices = set()
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

def calculate_potential_high_centers(leaf_contours, lon, lat, pressure):
    """计算潜在高压中心"""
    if lat[-1] < lat[0]:
        lat = lat[::-1]
        pressure = pressure[::-1, :]
    potential_centers = []
    seen_centers = set()
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

# ============================
# 单时间步处理函数
# ============================

def process_single_time(target_time, file_path_template='F:/ERA5/hourly/sfc/fin/ERA5_{hour}_sfc_{year}{month}_instant.nc'):
    """处理单个时间步的SLP数据并返回高压中心识别结果"""
    try:
        # 1. 数据读取与预处理
        year = target_time[:4]
        month = target_time[5:7]
        hour = target_time[-2:]  # 提取小时 (00, 06, 12, 18)
        hour_z = f"{hour}z"  # 转换为文件名中的格式 (00z, 06z, 12z, 18z)
        file_path = file_path_template.format(year=year, month=month, hour=hour_z)
        data = xr.open_dataset(file_path)
        data_sel = data.sel(valid_time=target_time)

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

        # 2. 绘制等压线
        interval = 2
        img_extent = [90, 270, 20, 90]
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection=ccrs.NorthPolarStereo(central_longitude=0))
        ax.set_extent(img_extent, crs=ccrs.PlateCarree())
        contours = plt.contour(
            lon_filtered, lat, pressure_above_1010,
            levels=np.arange(1010, np.nanmax(pressure_filtered), interval),
            transform=ccrs.PlateCarree(), colors='black'
        )
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
        min_radius_threshold = 1
        filtered_contours = [contour for contour in closed_contours if calculate_radius(contour) >= min_radius_threshold]

        # 5. 构建包含关系字典
        containment_dict = build_containment_dict(filtered_contours)

        # 6. 构建树结构
        trees = build_trees(containment_dict)
        for root in trees:
            update_depth(root)

        # 7. 树清洗逻辑
        cleaned_trees = clean_trees(trees)

        # 8. 提取潜在高压中心
        leaf_contours = extract_leaf_contours(cleaned_trees, filtered_contours)
        potential_high_centers = calculate_potential_high_centers(leaf_contours, lon_filtered, lat, pressure_above_1010)

        # 9. 将高压中心的经度重新映射回0到360范围
        for center in potential_high_centers:
            # 修正的重新映射逻辑: if >=180: -180 else +180
            center['lon'] = center['lon'] - 180 if center['lon'] >= 180 else center['lon'] + 180

        # 10. 准备CSV行数据
        date_str = target_time.replace('-', '').replace(':', '')[:12]
        csv_rows = []
        for center in potential_high_centers:
            csv_rows.append({
                'date': date_str,
                'center_lon': f"{center['lon']:.2f}",
                'center_lat': f"{center['lat']:.2f}",
                'stream': f"{center['stream']:.2f}"
            })
        return csv_rows
    except Exception as e:
        print(f"错误处理时间 {target_time}: {str(e)}")
        return []

# ============================
# 主程序
# ============================

if __name__ == "__main__":
    # 生成时间范围（例如，2020年1月1日到2020年12月31日，每6小时）
    start_date = '1940-01-01'
    end_date = '2024-12-31'
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    target_times = [f"{date.strftime('%Y-%m-%d')}-{hour:02d}" for date in dates for hour in [0, 6, 12, 18]]

    # 输出文件路径
    output_file = 'potential_high_pressure_centers_slp_0360.csv'

    # 按年份分组 target_times
    times_by_year = {}
    for target_time in target_times:
        year = target_time[:4]
        if year not in times_by_year:
            times_by_year[year] = []
        times_by_year[year].append(target_time)

    # 初始化 CSV 文件并写入表头
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['date', 'center_lon', 'center_lat', 'stream'])
        writer.writeheader()

        # 日志文件路径（统一保存）
    error_log_path = "error_log.txt"
    empty_log_path = "empty_results_log.txt"

    # 如果存在旧日志，先清空
    open(error_log_path, 'w', encoding='utf-8').close()
    open(empty_log_path, 'w', encoding='utf-8').close()

    # 按年循环处理
    # ============================
    for year in sorted(times_by_year.keys()):
        print(f"开始处理 {year} 年")
        year_times = times_by_year[year]
        results = {target_time: [] for target_time in year_times}

        # 暂存当年日志信息
        year_error_logs = []
        year_empty_logs = []

        # 使用多进程处理该年的时间步
        with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(process_single_time, target_time): target_time for target_time in year_times}
            # tqdm 进度条
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                               desc=f"处理 {year} 年时间步"):
                target_time = futures[future]
                try:
                    csv_rows = future.result()
                    results[target_time] = csv_rows
                except Exception as e:
                    msg = f"[{target_time}] {str(e)}"
                    print(f"错误处理时间 {target_time}: {str(e)}")
                    year_error_logs.append(msg)
                    results[target_time] = []

        # 按时间顺序写入该年的结果
        with open(output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['date', 'center_lon', 'center_lat', 'stream'])
            for target_time in year_times:
                if results[target_time]:
                    for row in results[target_time]:
                        writer.writerow(row)
                else:
                    year_empty_logs.append(f"[{target_time}] 无高压中心记录")

        # ✅ 每年写一次日志（集中写入总文件）
        if year_error_logs:
            with open(error_log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"\n===== {year} 年错误日志 =====\n")
                log_file.write("\n".join(year_error_logs) + "\n")

        if year_empty_logs:
            with open(empty_log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"\n===== {year} 年无高压中心 =====\n")
                log_file.write("\n".join(year_empty_logs) + "\n")

        print(f"完成 {year} 年，写入 CSV 与日志")
        results.clear()

    print(f"总运行时间: {time.time() - start:.2f} 秒")