import os
import csv

path = r'F:\ERA5\hourly\sfc\fin'  # 存放文件夹
hs = ['00','06','12','18']

missing = []
#ERA5_00z_sfc_194802.zip
for h in hs:
    for y in range(1940,2025):
        for m in range(1, 13):
            filename = f'ERA5_{h}z_sfc_{y}{str(m).zfill(2)}_instant.nc'
            filepath = os.path.join(path, filename)
            if not os.path.isfile(filepath):
                missing.append([h, y, m])
print(missing)
# 将缺失的文件记录写入 CSV 文件
output_csv = 'missing_files_sfc.csv'
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Hour', 'Year', 'Month'])  # 表头
    writer.writerows(missing)

print(f"已写入缺失文件列表至 {output_csv}")
