import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 读取Excel文件
data_path = 'E:/GEO/result/testdataset.xlsx'
data = pd.read_excel(data_path)
data = data.to_numpy()

# 选取后三列进行聚类
data_for_clustering = data[:, 8:]
'''[:, 2:]'''

# 标准化数据
scaler = StandardScaler()
data_for_clustering_scaled = scaler.fit_transform(data_for_clustering)

# 定义聚类数量的范围
min_clusters = 2
max_clusters = 12
n_clusters_range = range(min_clusters, max_clusters + 1)

# 存储每个聚类数量对应的聚类内部平方和
inertia_values = []

# 计算每个聚类数量对应的聚类内部平方和
for n_clusters in n_clusters_range:
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data_for_clustering_scaled)
    inertia_values.append(kmeans.inertia_)

# 绘制肘部法则曲线
plt.plot(n_clusters_range, inertia_values, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xticks(n_clusters_range)
plt.show()
