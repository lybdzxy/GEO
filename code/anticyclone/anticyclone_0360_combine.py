import pandas as pd
import os

def cluster_points(points, eps=5.0):
    """自定义聚类：经纬度差≤5度视为邻居"""
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

    final_points = []
    for cluster in clusters:
        if cluster:
            cluster_points = [points[i] for i in cluster]
            # For Northern Hemisphere, max stream; for Southern Hemisphere, min stream
            max_point = max(cluster_points, key=lambda p: p['stream']) if cluster_points[0]['lat'] > 0 else min(cluster_points, key=lambda p: p['stream'])
            final_points.append(max_point)
    return final_points

# 1. 读取两个CSV文件
try:
    df1 = pd.read_csv('potential_high_pressure_centers_stream_850hpa_0360_hemispheres.csv', encoding='utf-8')
    df2 = pd.read_csv('potential_high_pressure_centers_stream_850hpa_hemispheres.csv', encoding='utf-8')
    print("文件1和文件2已成功加载。")
    print("文件1行数：", len(df1))
    print("文件2行数：", len(df2))
except FileNotFoundError as e:
    print(f"错误：找不到文件 - {e}")
    exit()
except pd.errors.ParserError as e:
    print(f"CSV解析错误：{e}")
    exit()
except Exception as e:
    print(f"其他错误：{e}")
    exit()

# 2. 检查数据完整性
print("\n检查列名是否一致...")
if list(df1.columns) != list(df2.columns):
    print("错误：两个文件的列名不一致！")
    print("文件1列名：", df1.columns.tolist())
    print("文件2列名：", df2.columns.tolist())
    exit()
else:
    print("列名一致：", df1.columns.tolist())
    for col in df1.columns:
        if df1[col].dtype != df2[col].dtype:
            print(f"警告：列 '{col}' 的数据类型不一致！文件1: {df1[col].dtype}, 文件2: {df2[col].dtype}")

# 3. 检查键列的缺失值
key_columns = ['date', 'center_lon']
for df, name in [(df1, '文件1'), (df2, '文件2')]:
    if df[key_columns].isna().any().any():
        print(f"错误：{name} 中键列 {key_columns} 存在缺失值！")
        exit()

# 4. 查找文件2中缺少的数据
merge_df = pd.merge(df1, df2,
                    on=['date', 'center_lon'],
                    how='outer',
                    suffixes=('_file1', '_file2'),
                    indicator=True)

missing_in_file2 = merge_df[merge_df['_merge'] == 'left_only'].drop(columns=['_merge'], errors='ignore')

if len(missing_in_file2) > 0:
    print(f"\n发现文件2缺少 {len(missing_in_file2)} 条记录，这些记录将从文件1补充。")
    rename_cols = {col: col.replace('_file1', '') for col in missing_in_file2.columns if '_file1' in col}
    missing_in_file2 = missing_in_file2.rename(columns=rename_cols)
    missing_in_file2 = missing_in_file2[df2.columns]
    df2_updated = pd.concat([df2, missing_in_file2], ignore_index=True)
else:
    print("\n文件2中没有缺少文件1的数据。")
    df2_updated = df2.copy()

# 5. 去除重复数据
before_len = len(df2_updated)
df2_updated = df2_updated.drop_duplicates(subset=['date', 'center_lon', 'center_lat'])
after_len = len(df2_updated)
print(f"\n移除了 {before_len - after_len} 条重复记录（基于精确匹配）。")

# 6. 应用聚类方法进一步去重
print("\n应用聚类方法进一步去重...")
# 分组按日期处理
grouped = df2_updated.groupby('date')
clustered_data = []

for date, group in grouped:
    # 准备聚类输入
    points = group.rename(columns={'center_lon': 'lon', 'center_lat': 'lat', 'stream': 'stream'}).to_dict('records')
    # 应用聚类
    clustered_points = cluster_points(points, eps=5.0)
    # 转换回DataFrame
    clustered_group = pd.DataFrame(clustered_points).rename(columns={'lon': 'center_lon', 'lat': 'center_lat'})
    clustered_data.append(clustered_group)

# 合并所有日期的聚类结果
df2_updated = pd.concat(clustered_data, ignore_index=True)
print(f"聚类去重后剩余 {len(df2_updated)} 条记录。")

# 7. 按时刻和经度排序
df2_updated = df2_updated.sort_values(by=['date', 'center_lon'], ascending=[True, True]).reset_index(drop=True)

# 8. 保存合并后的数据
output_file = 'potential_high_pressure_centers_stream_850hpa_tot.csv'
if os.path.exists(output_file):
    print(f"警告：文件 {output_file} 已存在，将被覆盖。")
try:
    df2_updated.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n合并完成！结果已保存到 {output_file}")
    print(f"合并后总行数：{len(df2_updated)}")
except Exception as e:
    print(f"保存文件失败：{e}")
    exit()

# 9. 显示前几行结果
print("\n合并后数据的前5行（按时刻和经度排序）：")
print(df2_updated.head()[['date', 'center_lon', 'center_lat']].to_string(index=False))