import numpy as np
import matplotlib.pyplot as plt

#读取data.txt数据
data_path = f'E:/gis/Week05-OLS/data/data.txt'
data = np.loadtxt(data_path)

#看看数据长啥样
print(data)#整体情况
d1=data[0,:]
print(d1)#第一行情况（整体中间有几列被省略看不到）
#第一、二列没看明白，第七列应该是分组

#大致可以猜到第4、5、6列有一定相关性，画俩点图看看
plt.scatter(data[:,3],data[:,4], c=data[:, 6], s=0.1)
plt.show()
plt.scatter(data[:,3],data[:,5], c=data[:, 6], s=0.1)
plt.show()
#单独从点的分布可以看到四五列相关性比较好，四六列较差

#看看一二列的情况
plt.scatter(data[:,6],data[:,0])
plt.show()
plt.scatter(data[:,6],data[:,1])
plt.show()
#基于第七列的分组和第一二列的数据，第0、5、7、9、11可能是根据第一二列的数据数值大小进行分类的