import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from collections import Counter
from scipy.spatial.distance import cdist

# 读取Excel文件
data_path = f'E:/GEO/pyproject/trajectory_statistics.csv'
data = pd.read_csv(data_path)

# 选择 Y_bar < 70 的数据
filtered_data = data[data.iloc[:, 2] < 70]  # 假设 Y_bar 是第二列
# 选取数据进行聚类
data_for_clustering = filtered_data.iloc[:, 1:12].values

# 标准化数据
scaler = StandardScaler()
data_for_clustering_scaled = scaler.fit_transform(data_for_clustering)

# 定义聚类数量的范围
min_clusters = 2
max_clusters = 20

# 初始化存储指标的列表
sse = []
silhouette_scores = []
bic_scores = []
aic_scores = []
gap_scores = []
xie_beni_scores = []
dunn_scores = []


# Gap Statistic 计算函数
def gap_statistic(X, k_min, k_max):
    gaps = []
    for k in range(k_min, k_max + 1):
        # 对真实数据进行KMeans聚类
        kmeans = KMeans(n_clusters=k, random_state=100)
        kmeans.fit(X)
        Wk_real = np.sum(np.min(cdist(X, kmeans.cluster_centers_), axis=1))  # 聚类内的平方和

        # 生成与真实数据分布相似的随机数据
        random_data = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), X.shape)
        kmeans.fit(random_data)
        Wk_random = np.sum(np.min(cdist(random_data, kmeans.cluster_centers_), axis=1))  # 随机数据的平方和

        # 计算 Gap Statistic
        gap = np.log(Wk_random) - np.log(Wk_real)
        gaps.append(gap)

    return gaps


# 计算聚类
for n_clusters in range(min_clusters, max_clusters + 1):
    # 使用KMeans算法进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=100)
    kmeans.fit(data_for_clustering_scaled)

    # 计算总误差平方和（SSE）
    sse.append(kmeans.inertia_)

    # 获取聚类结果的标签
    labels = kmeans.labels_

    # 计算轮廓系数
    silhouette_avg = silhouette_score(data_for_clustering_scaled, labels)
    silhouette_scores.append(silhouette_avg)

    # 使用GaussianMixture模型计算BIC和AIC
    gmm = GaussianMixture(n_components=n_clusters, random_state=100)
    gmm.fit(data_for_clustering_scaled)
    bic_scores.append(gmm.bic(data_for_clustering_scaled))
    aic_scores.append(gmm.aic(data_for_clustering_scaled))

    # 计算Xie-Beni指数
    xie_beni = np.min(cdist(kmeans.cluster_centers_, kmeans.cluster_centers_)) / np.mean(
        np.min(cdist(data_for_clustering_scaled, kmeans.cluster_centers_), axis=1))
    xie_beni_scores.append(xie_beni)

    # 计算Dunn指数
    intra_cluster_dist = np.mean(
        [np.min(cdist(data_for_clustering_scaled[labels == i], data_for_clustering_scaled[labels == i])) for i in
         range(n_clusters)])
    inter_cluster_dist = np.min(cdist(kmeans.cluster_centers_, kmeans.cluster_centers_))
    dunn_index = inter_cluster_dist / intra_cluster_dist
    dunn_scores.append(dunn_index)

# 计算Gap Statistic
gap_scores = gap_statistic(data_for_clustering_scaled, min_clusters, max_clusters)

# 绘制SSE图（肘部法则）
plt.plot(range(min_clusters, max_clusters + 1), sse, marker='o')
plt.xlabel('聚类数')
plt.ylabel('SSE')
plt.title('SSE - 肘部法则')
plt.show()

# 绘制BIC/AIC图
plt.plot(range(min_clusters, max_clusters + 1), bic_scores, label='BIC')
plt.plot(range(min_clusters, max_clusters + 1), aic_scores, label='AIC')
plt.xlabel('聚类数')
plt.ylabel('BIC/AIC')
plt.title('BIC/AIC 方法')
plt.legend()
plt.show()

# 绘制Gap Statistic图
plt.plot(range(min_clusters, max_clusters + 1), gap_scores, marker='o')
plt.xlabel('聚类数')
plt.ylabel('Gap Statistic')
plt.title('Gap Statistic 方法')
plt.show()

# 绘制轮廓系数图
plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker='o')
plt.xlabel('聚类数')
plt.ylabel('轮廓系数')
plt.title('轮廓系数 方法')
plt.show()

# 绘制Xie-Beni图
plt.plot(range(min_clusters, max_clusters + 1), xie_beni_scores, marker='o')
plt.xlabel('聚类数')
plt.ylabel('Xie-Beni Index')
plt.title('Xie-Beni 指数')
plt.show()

# 绘制Dunn Index图
plt.plot(range(min_clusters, max_clusters + 1), dunn_scores, marker='o')
plt.xlabel('聚类数')
plt.ylabel('Dunn Index')
plt.title('Dunn Index 方法')
plt.show()

# 综合考虑多个指标，选择最佳聚类数
best_n_clusters = {
    'SSE': np.argmin(sse) + min_clusters,
    '轮廓系数': np.argmax(silhouette_scores) + min_clusters,
    'Gap Statistic': np.argmax(gap_scores) + min_clusters,
    'BIC': np.argmin(bic_scores) + min_clusters,
    'AIC': np.argmin(aic_scores) + min_clusters,
    'Xie-Beni': np.argmin(xie_beni_scores) + min_clusters,
    'Dunn Index': np.argmax(dunn_scores) + min_clusters
}

# 选择最常出现的最佳聚类数
best_cluster_count = Counter(best_n_clusters.values()).most_common(1)[0][0]
print(f"综合考虑多个指标，选择的最佳聚类数: {best_cluster_count}")

# 使用最优的聚类数量进行聚类
best_kmeans = KMeans(n_clusters=best_cluster_count, random_state=100)
filtered_data["cluster"] = best_kmeans.fit_predict(data_for_clustering_scaled)

# 保存聚类结果
filtered_data.to_csv(f'E:/GEO/pyproject/k-means_new.csv', index=False)
