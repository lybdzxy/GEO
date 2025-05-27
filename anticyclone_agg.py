import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# 读取CSV文件
data_path = 'E:/GEO/pyproject/trajectory_statistics.csv'
data = pd.read_csv(data_path)

# 选择 Y_bar < 70 的数据
filtered_data = data[data.iloc[:, 2] < 70].copy()  # 假设 Y_bar 是第二列

# 选取用于聚类的列（X_bar, Y_bar, Var_x, Var_y）
data_for_clustering = filtered_data.iloc[:, 1:5].values

# 标准化数据
scaler = StandardScaler()
data_for_clustering_scaled = scaler.fit_transform(data_for_clustering)

# 设定层次聚类，设定聚类数（例如 6 类）
n_clusters = 6
agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
filtered_data.loc[:, "cluster"] = agg_clustering.fit_predict(data_for_clustering_scaled)

# 保存聚类结果
output_path = 'E:/GEO/pyproject/hierarchical_clustering_result.csv'
filtered_data.to_csv(output_path, index=False)

print(f"层次聚类完成，结果已保存至 {output_path}")

# 画出层次聚类的树状图
plt.figure(figsize=(12, 6))
Z = linkage(data_for_clustering_scaled, method='ward')
dendrogram(Z)
plt.title("Dendrogram of Agglomerative Clustering")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()
