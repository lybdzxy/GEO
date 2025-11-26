import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import os
from sklearn.metrics import pairwise_distances
from sklearn.utils import resample
import time

time = time.time()

def gap_statistic(data, n_clusters, n_bootstraps=100):
    # 计算聚类效果
    model = fcluster(linkage(data, 'ward'), n_clusters, criterion='maxclust')
    original_dispersion = np.sum(pairwise_distances(data, metric='euclidean') ** 2) / 2

    # 计算基于均匀分布的预期聚合度
    bootstrapped_dispersion = []
    for _ in range(n_bootstraps):
        # 从数据中自助采样
        sample = resample(data, n_samples=data.shape[0], replace=True)
        model_sample = fcluster(linkage(sample, 'ward'), n_clusters, criterion='maxclust')
        bootstrapped_dispersion.append(np.sum(pairwise_distances(sample, metric='euclidean') ** 2) / 2)

    # 计算gap statistic
    expected_dispersion = np.mean(bootstrapped_dispersion)
    gap = np.log(expected_dispersion) - np.log(original_dispersion)

    return gap

try:
    # 读取Excel文件
    data_path = 'E:/GEO/etccdi/coor_pc1.xlsx'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")

    data = pd.read_excel(data_path)
    data = data.to_numpy()

    # 选取用于聚类的列
    data_for_clustering = data[:, :3]

    # 标准化数据
    scaler = StandardScaler()
    data_for_clustering_scaled = scaler.fit_transform(data_for_clustering)

    # 使用层次聚类的链接矩阵
    Z = linkage(data_for_clustering_scaled, 'ward')

    # 绘制树形图
    plt.figure(figsize=(10, 7))
    dendrogram(Z)
    plt.title('Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.show()

    # 定义聚类数量的范围
    min_clusters = 2
    max_clusters = 12

    # 计算每个聚类数量的gap statistic
    gap_scores = []
    for n_clusters in range(min_clusters, max_clusters + 1):
        gap = gap_statistic(data_for_clustering_scaled, n_clusters)
        gap_scores.append(gap)
        print(f'Clusters: {n_clusters}, Gap Statistic: {gap:.4f}')

    # 找到最大gap statistic对应的聚类数量
    best_n_clusters = np.argmax(gap_scores) + min_clusters
    print(f'Best number of clusters: {best_n_clusters}')

    # 使用最优的聚类数量进行最终的聚类
    labels = fcluster(Z, best_n_clusters, criterion='maxclust')

    # 将聚类标签添加到原数据中
    data_with_labels = np.column_stack((data_for_clustering, labels))
    # 转换为 DataFrame
    result_xl = pd.DataFrame(data_with_labels)

    # 重新命名列
    result_xl.columns = ['lon', 'lat', 'pc1', 'cluster']

    # 将结果保存为Excel文件
    result_xl.to_excel(f'E:/GEO/result/ecm/Agglomerative_gap{time}.xlsx', index=False)
    result = {}
    for i in data_with_labels:
        lon, lat, pc1, cluster = i
        result[(lon, lat)] = (pc1, cluster)

    result_df = pd.DataFrame(result).T  # 转置数据

    # 重新命名列
    result_df.columns = ['pc1', 'cluster']

    # 确保保存路径存在
    save_path = 'E:/GEO/result/ecm'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 将结果保存为NetCDF文件
    result_df.to_xarray().to_netcdf(os.path.join(save_path, f'Agglomerative_gap{time}.nc'))

except FileNotFoundError as fnf_error:
    print(f"File error: {fnf_error}")
except Exception as e:
    print(f'Error processing: {e}')
