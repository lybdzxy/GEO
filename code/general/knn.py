import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


# 加载数据
data_path = 'E:/GEO/result/ecm/agglomerative_gap.xlsx'
data = pd.read_excel(data_path)

# 将数据转换为numpy数组
data = data.to_numpy()

# 定义特征和标签
X = data[:, :2]  # 前两列作为特征
y = data[:, 3]  # 第11列作为标签

# 定义KNN分类器并拟合数据
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X, y)

# 绘制决策边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

lon = xx.ravel().T
lat = yy.ravel().T
cluster = Z.reshape(xx.shape)

result_data = np.column_stack((lon,lat,cluster.ravel()))

result = {}
for i in result_data:
    lon, lat, cluster = i
    result[(lon, lat)] = (cluster,cluster)
result_df = pd.DataFrame(result).T  # 转置数据

# 重新命名列
result_df.columns = ['cluster','back']

# 将结果保存为NetCDF文件
result_df.to_xarray().to_netcdf('E:/GEO/result/ecm/pca_hcc_knn.nc')


'''plt.contourf(xx, yy, Z, alpha=1, cmap='Pastel1')  # 调整决策边界的颜色
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k', cmap='Pastel2')  # 调整散点图的颜色
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary of KNN')
plt.show()
'''