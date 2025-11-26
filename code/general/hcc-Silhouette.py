import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import os

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

    # 初始化最优轮廓系数和对应的聚类数量
    best_score = -1
    best_n_clusters = min_clusters

    # 循环尝试不同的聚类数量
    from sklearn.metrics import silhouette_score

    for n_clusters in range(min_clusters, max_clusters + 1):
        # 使用scipy的fcluster方法获取聚类标签
        labels = fcluster(Z, n_clusters, criterion='maxclust')

        # 计算轮廓系数
        silhouette_avg = silhouette_score(data_for_clustering_scaled, labels)
        print(f'Clusters: {n_clusters}, Silhouette Score: {silhouette_avg:.4f}')

        # 如果当前轮廓系数更高，则更新最优值
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_n_clusters = n_clusters

    # 使用最优的聚类数量进行最终的聚类
    labels = fcluster(Z, best_n_clusters, criterion='maxclust')

    # 将聚类标签添加到原数据中
    data_with_labels = np.column_stack((data_for_clustering, labels))
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
    result_df.to_xarray().to_netcdf(os.path.join(save_path, 'Agglomerative.nc'))

except FileNotFoundError as fnf_error:
    print(f"File error: {fnf_error}")
except Exception as e:
    print(f'Error processing: {e}')
