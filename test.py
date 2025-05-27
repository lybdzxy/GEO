import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from collections import Counter

# 读取CSV文件
data_path = 'E:/GEO/pyproject/trajectory_statistics.csv'
data = pd.read_csv(data_path)

# 选择 Y_bar < 70 的数据
filtered_data = data[data.iloc[:, 2] < 70]

# 第一次聚类：对第二列和第三列进行聚类
data_for_first_clustering = filtered_data.iloc[:, 4:6].values

# 标准化数据
scaler_first = StandardScaler()
data_for_first_clustering_scaled = scaler_first.fit_transform(data_for_first_clustering)

# 定义聚类数量的范围
min_clusters = 2
max_clusters = 10

# 初始化存储指标的列表
sse = []
silhouette_scores = []
db_scores = []
ch_scores = []

# 循环尝试不同的聚类数量
for n_clusters in range(min_clusters, max_clusters + 1):
    # 使用KMeans算法进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=100)
    kmeans.fit(data_for_first_clustering_scaled)

    # 计算总误差平方和（SSE）
    sse.append(kmeans.inertia_)

    # 获取聚类结果的标签
    labels = kmeans.labels_

    # 计算轮廓系数
    silhouette_avg = silhouette_score(data_for_first_clustering_scaled, labels)
    silhouette_scores.append(silhouette_avg)

    # 计算Davies-Bouldin指数
    db_index = davies_bouldin_score(data_for_first_clustering_scaled, labels)
    db_scores.append(db_index)

    # 计算Calinski-Harabasz指数
    ch_index = calinski_harabasz_score(data_for_first_clustering_scaled, labels)
    ch_scores.append(ch_index)

# 确定最佳聚类数
best_n_clusters = {
    'SSE': np.argmin(sse) + min_clusters,
    '轮廓系数': np.argmax(silhouette_scores) + min_clusters,
    'Davies-Bouldin指数': np.argmin(db_scores) + min_clusters,
    'Calinski-Harabasz指数': np.argmax(ch_scores) + min_clusters
}

print(f"基于SSE的最佳聚类数: {best_n_clusters['SSE']}")
print(f"基于轮廓系数的最佳聚类数: {best_n_clusters['轮廓系数']}")
print(f"基于Davies-Bouldin指数的最佳聚类数: {best_n_clusters['Davies-Bouldin指数']}")
print(f"基于Calinski-Harabasz指数的最佳聚类数: {best_n_clusters['Calinski-Harabasz指数']}")

# 综合考虑多个指标，选择最佳聚类数
# 这里采用简单的投票机制，选择出现次数最多的聚类数
best_cluster_count = Counter(best_n_clusters.values()).most_common(1)[0][0]
print(f"综合考虑多个指标，选择的最佳聚类数: {best_cluster_count}")

# 使用最佳聚类数进行第一次聚类
final_kmeans_first = KMeans(n_clusters=best_cluster_count, random_state=100)
filtered_data["cluster_first"] = final_kmeans_first.fit_predict(data_for_first_clustering_scaled)

# 第二次聚类：在第一次聚类结果基础上，对第四至第八列进行聚类
min_clusters_second = 2
max_clusters_second = 10

# 获取第一次聚类的唯一簇标签
unique_clusters_first = filtered_data["cluster_first"].unique()

# 创建一个空的列表，用于存储第二次聚类的结果
second_cluster_labels = []

# 遍历每个第一次聚类的簇
for cluster in unique_clusters_first:
    # 获取属于当前簇的数据
    cluster_data = filtered_data[filtered_data["cluster_first"] == cluster]
    data_for_second_clustering = cluster_data.iloc[:, [1, 2, 6, 7]].values

    # 标准化数据
    scaler_second = StandardScaler()
    data_for_second_clustering_scaled = scaler_second.fit_transform(data_for_second_clustering)

    # 初始化存储指标的列表
    sse_second = []
    silhouette_scores_second = []
    db_scores_second = []
    ch_scores_second = []

    # 循环尝试不同的聚类数量
    for n_clusters in range(min_clusters_second, max_clusters_second + 1):
        # 使用KMeans算法进行聚类
        kmeans_second = KMeans(n_clusters=n_clusters, random_state=100)
        kmeans_second.fit(data_for_second_clustering_scaled)

        # 计算总误差平方和（SSE）
        sse_second.append(kmeans_second.inertia_)

        # 获取聚类结果的标签
        labels_second = kmeans_second.labels_

        # 计算轮廓系数
        silhouette_avg_second = silhouette_score(data_for_second_clustering_scaled, labels_second)
        silhouette_scores_second.append(silhouette_avg_second)

        # 计算Davies-Bouldin指数
        db_index_second = davies_bouldin_score(data_for_second_clustering_scaled, labels_second)
        db_scores_second.append(db_index_second)

        # 计算Calinski-Harabasz指数
        ch_index_second = calinski_harabasz_score(data_for_second_clustering_scaled, labels_second)
        ch_scores_second.append(ch_index_second)

    # 确定最佳聚类数
    best_n_clusters_second = {
        'SSE': np.argmin(sse_second) + min_clusters_second,
        '轮廓系数': np.argmax(silhouette_scores_second) + min_clusters_second,
        'Davies-Bouldin指数': np.argmin(db_scores_second) + min_clusters_second,
        'Calinski-Harabasz指数': np.argmax(ch_scores_second) + min_clusters_second
    }

    print(f"簇 {cluster} - 基于SSE的最佳聚类数: {best_n_clusters_second['SSE']}")
    print(f"簇 {cluster} - 基于轮廓系数的最佳聚类数: {best_n_clusters_second['轮廓系数']}")
    print(f"簇 {cluster} - 基于Davies-Bouldin指数的最佳聚类数: {best_n_clusters_second['Davies-Bouldin指数']}")
    print(f"簇 {cluster} - 基于Calinski-Harabasz指数的最佳聚类数: {best_n_clusters_second['Calinski-Harabasz指数']}")

    # 综合考虑多个指标，选择最佳聚类数
    best_cluster_count_second = Counter(best_n_clusters_second.values()).most_common(1)[0][0]
    print(f"簇 {cluster} - 综合考虑多个指标，选择的最佳聚类数: {best_cluster_count_second}")

    # 使用最佳聚类数进行第二次聚类
    final_kmeans_second = KMeans(n_clusters=best_cluster_count_second, random_state=100)
    second_labels = final_kmeans_second.fit_predict(data_for_second_clustering_scaled)

    # 将第二次聚类的标签添加到列表中
    second_cluster_labels.extend(second_labels)

# 将第二次聚类的结果添加到原始数据中
filtered_data["cluster_second"] = second_cluster_labels

# 将结果保存到新的CSV文件
filtered_data.to_csv('E:/GEO/pyproject/k-means_multi.csv', index=False)
