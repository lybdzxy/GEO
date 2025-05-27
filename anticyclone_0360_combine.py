import pandas as pd

# 1. 读取两个CSV文件
try:
    # 假设文件名为 file1.csv 和 file2.csv
    df1 = pd.read_csv('output_filtered_0360.csv')  # 文件1
    df2 = pd.read_csv('output_filtered.csv')  # 文件2

    print("文件1和文件2已成功加载。")
    print("文件1行数：", len(df1))
    print("文件2行数：", len(df2))

except FileNotFoundError as e:
    print(f"错误：找不到文件 - {e}")
    exit()
except Exception as e:
    print(f"发生错误：{e}")
    exit()

# 2. 检查数据完整性
print("\n检查列名是否一致...")
if list(df1.columns) != list(df2.columns):
    print("警告：两个文件的列名不一致！")
    print("文件1列名：", df1.columns.tolist())
    print("文件2列名：", df2.columns.tolist())
else:
    print("列名一致：", df1.columns.tolist())

# 3. 假设唯一标识是 'date' 和 'center_lon' 的组合，查找文件2中缺少的数据
# 合并两个DataFrame，标记来源
df1['source'] = 'file1'
df2['source'] = 'file2'

# 按 'date' 和 'center_lon' 合并，找出文件2中没有的数据
merge_df = pd.merge(df1, df2,
                    on=['date', 'center_lon', 'center_lat', 'pressure', 'contour_points_x', 'contour_points_y'],
                    how='outer',
                    suffixes=('_file1', '_file2'),
                    indicator=True)

# 找出只在文件1中的数据（文件2缺少的部分）
missing_in_file2 = merge_df[merge_df['_merge'] == 'left_only'].drop(columns='_merge')

if len(missing_in_file2) > 0:
    print(f"\n发现文件2缺少 {len(missing_in_file2)} 条记录，这些记录将从文件1补充。")
    # 将缺失的数据添加到文件2
    df2_updated = pd.concat([df2, missing_in_file2.drop(columns='source_file1')], ignore_index=True)
else:
    print("\n文件2中没有缺少文件1的数据。")
    df2_updated = df2.copy()

# 4. 去除可能的重复数据（如果有）
df2_updated = df2_updated.drop_duplicates(subset=['date', 'center_lon', 'center_lat'])

# 5. 保存合并后的数据到新CSV文件
output_file = 'output_filtered_tot.csv'
df2_updated.to_csv(output_file, index=False)

print(f"\n合并完成！结果已保存到 {output_file}")
print(f"合并后总行数：{len(df2_updated)}")

# 6. 额外检查：显示前几行结果
print("\n合并后数据的前5行：")
print(df2_updated.head().to_string())
