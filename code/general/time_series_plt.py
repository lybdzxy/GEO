import pandas as pd
import matplotlib.pyplot as plt

districts = ['E', 'N', 'NC', 'NE', 'S', 'SC', 'W']
indices = ['prcptot', 'r95p', 'r99p', 'sdii', 'rx1day']

for district in districts:
    for index in indices:
        data_126path = f'E:/GEO/result/ecm/{district}_126_{index}.xlsx'
        data_126 = pd.read_excel(data_126path)
        data_126 = data_126.sort_values(by='year')
        pre_126 = data_126['mean_value'].to_numpy()
        year_126 = data_126['year'].to_numpy()

        data_585path = f'E:/GEO/result/ecm/{district}_585_{index}.xlsx'
        data_585 = pd.read_excel(data_585path)
        data_585 = data_585.sort_values(by='year')
        pre_585 = data_585['mean_value'].to_numpy()
        year_585 = data_585['year'].to_numpy()

        data_obspath = f'E:/GEO/result/ecm/{district}_obs_{index}.xlsx'
        data_obs = pd.read_excel(data_obspath)
        data_obs = data_obs.sort_values(by='year')
        pre_obs = data_obs['mean_value'].to_numpy()
        year_obs = data_obs['year'].to_numpy()

        plt.figure(figsize=(12,6))

        # 绘制数据趋势图
        plt.plot(year_126, pre_126, 'b-', label='SSP 126')
        plt.plot(year_585, pre_585, 'r-', label='SSP 585')
        plt.plot(year_obs, pre_obs, 'k-', label='His')

        # 添加标签、标题和图例
        plt.xlabel('Year')
        plt.ylabel(index)
        plt.legend()

        plt.savefig(f'E:/GEO/result/ecm/pic/{district}{index}.png',dpi=600)
        plt.close()