import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

indices = ['prcptot', 'r95p', 'r99p', 'sdii', 'rx1day']
ssps = [126, 585]
for ssp in ssps:
    for index in indices:
        # 读取 Excel 文件
        data_path = f'E:/GEO/result/ecm/tos_{index}_{ssp}.xlsx'
        data = pd.read_excel(data_path)

        # 将 'time' 列转换为 datetime 对象
        data['time'] = pd.to_datetime(data['time'])

        # 创建图表
        plt.figure(figsize=(12, 6))
        plt.plot(data['time'], data['le_mode_0'])

        # 设置X轴的日期格式，只显示年份
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # 添加X轴和Y轴标签，图表标题，以及网格
        '''plt.xlabel('Year')
        plt.ylabel('re_mode_0')
        plt.title('Time Series of re_mode_0')'''
        plt.grid(True)

        # 显示图表
        plt.savefig(f'E:/GEO/result/ecm/pic/tos_{index}_{ssp}_ts.png', dpi=600, bbox_inches='tight')
        plt.close()
