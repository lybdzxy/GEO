import netCDF4 as nc
import numpy as np

# 打开NetCDF文件
nc_file = nc.Dataset('E:/test/t/b.nc', 'r')

# 定义要提取的日期，这里假设日期格式为'YYYY-MM-DD'
target_date = '1951-01-01'

# 提取对应时间步的栅格数据
# 假设时间维度名为'time'，栅格变量名为'pr'
# 根据时间值筛选出指定日期的数据
time_var = nc_file.variables['time']
pr_var = nc_file.variables['pr']

# 将日期字符串转换为时间值，前提是NetCDF文件中的时间值也是以相同的格式存储的
target_time = np.datetime64(target_date)

# 找到与目标日期最接近的时间步索引
# 这里假设时间步是按照升序排列的
# 需要将时间维度的值也转换为与目标时间相同的数据类型
time_values = time_var[:]
closest_time_index = np.argmin(np.abs(time_values.astype('datetime64[D]') - target_time))

# 提取对应时间步的栅格数据
grid_data = pr_var[closest_time_index]

# 计算栅格中的最大值
max_value = np.min(grid_data)

# 输出最大值
print(f'The maximum value on {target_date} is {max_value}')

# 关闭NetCDF文件
nc_file.close()
