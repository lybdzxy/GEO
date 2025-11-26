import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import skfuzzy as fuzz

def cluster_data(index):
    try:
        # 读取Excel文件
        data_path = f'E:/GEO/result/ecm/obs{index}_lom_GEV.xlsx'
        data = pd.read_excel(data_path)
        data = data.to_numpy()

        # 选取用于聚类的列
        data_for_clustering = data

        # 标准化数据
        scaler = StandardScaler()
        data_for_clustering_scaled = scaler.fit_transform(data_for_clustering)

        # 定义聚类数量的范围
        min_clusters = 2
        max_clusters = 12

        # 初始化最优轮廓系数和对应的聚类数量
        best_score = -1
        best_n_clusters = min_clusters

        # 循环尝试不同的聚类数量
        for n_clusters in range(min_clusters, max_clusters + 1):
            # 使用Fuzzy C-Means算法进行聚类
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                data_for_clustering_scaled.T, n_clusters, 2, error=0.005, maxiter=1000, init=None, seed=42
            )

            # 计算模糊分割系数（FPC）
            silhouette_avg = fpc
            print(f'Index: {index}, Clusters: {n_clusters}, Fuzzy Partition Coefficient: {silhouette_avg:.4f}')

            # 如果当前模糊分割系数更高，则更新最优值
            if silhouette_avg > best_score:
                best_score = silhouette_avg
                best_n_clusters = n_clusters

        # 使用最优的聚类数量进行最终的Fuzzy C-Means聚类
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data_for_clustering_scaled.T, best_n_clusters, 2, error=0.005, maxiter=1000, init=None, seed=42
        )

        # 获取硬分类标签
        labels = np.argmax(u, axis=0)

        # 将聚类标签添加到原数据中
        data_with_labels = np.column_stack((data, labels))
        result = {}
        for i in data_with_labels:
            lon, lat, loc, scale, shape, cluster = i
            result[(lon, lat)] = (loc, scale, shape, cluster)
        result_df = pd.DataFrame(result).T  # 转置数据

        # 重新命名列
        result_df.columns = ['loc', 'scale', 'shape', 'cluster']

        # 将结果保存为NetCDF文件
        result_df.to_xarray().to_netcdf(f'E:/GEO/result/ecm/obs{index}_fcm.nc')
        print(f'Results for {index} saved successfully.')

    except Exception as e:
        print(f'Error processing {index}: {e}')

# 处理多个索引
for index in ('rx1day', 'r95p', 'r99p'):
    cluster_data(index)
