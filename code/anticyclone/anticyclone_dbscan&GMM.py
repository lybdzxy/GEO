import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# 读取数据
data_path = 'E:/GEO/pyproject/trajectory_statistics.csv'
data = pd.read_csv(data_path)

# 选取后三列进行聚类
data_for_clustering = data.iloc[:, 1:5].values

# 标准化数据
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_for_clustering)

# ----------- 通过 K-距离图选择 min_samples 和 eps -----------

# 设置选择的邻居数 k
k = 50  # 设置你认为合适的 k 值，例如 5
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(data_scaled)

# 计算每个点到其第 k 个邻居的距离
distances, indices = neighbors_fit.kneighbors(data_scaled)

# 对距离进行排序，取最后一个邻居的距离（第 k 个最近邻）
distances = np.sort(distances[:, -1])

# 画 K-距离图
plt.plot(distances)
plt.xlabel('Data Points')
plt.ylabel(f'Distance to {k}th Nearest Neighbor')
plt.title('K-distance Graph')
plt.show()

# 根据 K-距离图选择合适的 eps（假设通过观察图选择了 eps=2.5）
optimal_eps = 2.5  # 根据图选择的合适的 eps 值
optimal_min_samples = k  # 这里我们选择 k 为 min_samples

# ----------- DBSCAN 聚类 -----------

# 使用 DBSCAN 进行聚类
dbscan = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples, metric='manhattan')
labels_dbscan = dbscan.fit_predict(data_scaled)

# 将 DBSCAN 聚类标签添加到数据中
data['cluster_dbscan'] = labels_dbscan

# ----------- GMM 聚类 -----------

# 确定 GMM 最优 n_components（选择合适的范围，例如2到15）
bic_scores = []
sil_scores = []
components_range = range(2, 15)

for n in components_range:
    gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=100)
    labels = gmm.fit_predict(data_scaled)
    bic_scores.append(gmm.bic(data_scaled))
    sil_scores.append(silhouette_score(data_scaled, labels))

# 选择最优 n_components，结合 BIC 和轮廓系数
optimal_n_components = components_range[np.argmin(bic_scores)]
if max(sil_scores) > 0.2:  # 设定最低轮廓系数阈值，避免过度分类
    optimal_n_components = components_range[np.argmax(sil_scores)]

# 使用最优 n_components 进行 GMM 聚类
gmm = GaussianMixture(n_components=optimal_n_components, covariance_type='diag', random_state=100)
labels_gmm = gmm.fit_predict(data_scaled)

# 将 GMM 聚类标签添加到数据中
data['cluster_gmm'] = labels_gmm

# ----------- 保存结果 -----------

# 保存最终的聚类结果到 CSV 文件
data.to_csv('E:/GEO/pyproject/dbscan_gmm_results_coor.csv', index=False)

print(f"DBSCAN 和 GMM 聚类完成，最佳 eps={optimal_eps}, min_samples={optimal_min_samples}, n_components={optimal_n_components}，结果已保存。")
