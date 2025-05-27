import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
# 假设文件路径为 'data.csv'
df = pd.read_csv('trajectories_filtered.csv')

# 绘制气压中心的累计频数分布图
plt.figure(figsize=(10, 6))
sns.histplot(df['center_pressure'], kde=True, cumulative=True, bins=30, color='blue')

# 设置图表标题和标签
plt.title('北半球温带反气旋气压中心累计频数分布图')
plt.xlabel('气压 (hPa)')
plt.ylabel('累计频数')

# 显示图表
plt.show()
