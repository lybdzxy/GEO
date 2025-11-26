import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genextreme
import pandas as pd

params_path = 'E:/GEO/result/new/GEV.xlsx'
df = pd.read_excel(params_path)

# 将index、ssp和district列设置为索引
df.set_index(['index', 'ssp', 'district'], inplace=True)

indices = ['r95p','r99p','rx1day']
ssps = ['126','585','his']
districts = ['NW','NE','SE']

# 自定义不同ssp的颜色
ssp_colors = {
    '126': 'b',
    '585': 'r',
    'his': 'k'
}

for index in indices:
    if index == 'r95p':
        x_values = np.linspace(0, 2000, 1000)
    elif index == 'r99p':
        x_values = np.linspace(0, 1000, 1000)
    elif index == 'rx1day':
        x_values = np.linspace(0, 500, 1000)


    for district in districts:
        # 创建一个图
        plt.figure(figsize=(12,9))
        for ssp in ssps:
            data = df.loc[(index, ssp, district), ['loc', 'scale', 'shape']]
            pdf_values = genextreme.pdf(x_values, data['shape'], loc=data['loc'], scale=data['scale'])
            # 使用颜色字典中对应的颜色
            plt.plot(x_values, pdf_values, color=ssp_colors[ssp], label=f'{ssp}')

        # 添加标签、标题和图例
        plt.xlabel('Precipitation')
        plt.ylabel('Probability Density')
        plt.title(f'{index}, {district}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'E:/GEO/result/new/{index}_{district}_GEV.png', dpi=600,bbox_inches='tight')
        plt.close()  # 关闭当前图形，以便在下一次迭代中创建新的图形
