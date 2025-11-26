import matplotlib.pyplot as plt  # 绘图库
from matplotlib import rcParams  # 用于定制绘图参数的模块
import numpy as np  # 数值计算库
import skill_metrics as sm  # 自定义模块用于计算技能指标
import pandas as pd

indices = ['prcptot', 'r95p', 'r99p', 'r95ptot', 'r99ptot', 'sdii', 'rx1day', 'r10mm', 'r20mm']

# 设置图形属性
rcParams["figure.figsize"] = [8.0, 6.4]  # 设置图形尺寸
rcParams['lines.linewidth'] = 1  # 设置绘图线宽
rcParams.update({'font.size': 12})  # 设置坐标轴文本的字体大小
["#b4ddd4", "#075c62", "#52dcbc", "#1e7b20", "#88dc40", "#743502", "#da7642", "#be0332", "#ebb9c7", "#ee77a0", "#9d7b84", "#fcce6a"]
MARKERS = {
    "EC-Earth3": {
        "labelColor": "k",
        "symbol": "+",
        "size": 9,
        "faceColor": "#b4ddd4",
        "edgeColor": "#b4ddd4",
    },
    "MPI-ESM1-2-HR": {
        "labelColor": "k",
        "symbol": ".",
        "size": 9,
        "faceColor": "#075c62",
        "edgeColor": "#075c62",
    },
    "EC-Earth3-Veg": {
        "labelColor": "k",
        "symbol": "x",
        "size": 9,
        "faceColor": "#52dcbc",
        "edgeColor": "#52dcbc",
    },
    "CMCC-ESM2": {
        "labelColor": "k",
        "symbol": "s",
        "size": 9,
        "faceColor": "#1e7b20",
        "edgeColor": "#1e7b20",
    },
    "GFDL-ESM4": {
        "labelColor": "k",
        "symbol": "d",
        "size": 9,
        "faceColor": "#88dc40",
        "edgeColor": "#88dc40",
    },
    "INM-CM4-8": {
        "labelColor": "k",
        "symbol": "^",
        "size": 9,
        "faceColor": "#743502",
        "edgeColor": "#743502",
    },
    "BCC-CSM2-MR": {
        "labelColor": "k",
        "symbol": "v",
        "size": 9,
        "faceColor": "#da7642",
        "edgeColor": "#da7642",
    },
    "MRI-ESM2-0": {
        "labelColor": "k",
        "symbol": "p",
        "size": 9,
        "faceColor": "#be0332",
        "edgeColor": "#be0332",
    },
    "CMCC-CM2-SR5": {
        "labelColor": "k",
        "symbol": "h",
        "size": 9,
        "faceColor": "#ebb9c7",
        "edgeColor": "#ebb9c7",
    },
    "TaiESM1": {
        "labelColor": "k",
        "symbol": "*",
        "size": 9,
        "faceColor": "#ee77a0",
        "edgeColor": "#ee77a0",
    },
    "NorESM2-MM": {
        "labelColor": "k",
        "symbol": "H",
        "size": 9,
        "faceColor": "#9d7b84",
        "edgeColor": "#9d7b84",
    },
    "INM-CM5-0": {
        "labelColor": "k",
        "symbol": "8",
        "size": 9,
        "faceColor": "#fcce6a",
        "edgeColor": "#fcce6a",
    },
}

for index in indices:
    # 读取Excel文件
    data = pd.read_excel("E:/GEO/etccdi/qpm/model_indices_summary - 副本.xlsx",sheet_name=index)
    ref = pd.read_excel("E:/GEO/etccdi/qpm/ref.xlsx")

    # 提取需要的列
    models = data['Model'].unique().tolist()  # 将模型名称转换为列表
    years = data['Year'].unique()
    sdev = []
    crmsd_all = []
    ccoef = []
    crmsd = []
    ref_data = ref[ref['Index']==index]
    ref_st = ref_data['Std_Dev_Predicted'].iloc[0]
    for model in models:
        model_data = data[data['Model'] == model]
        crmsd_all.append(model_data['RMSE'].iloc[0])  # 中心化 RMSD
    crmsd_max = max(crmsd_all)
    # 计算每个模型的泰勒统计量
    for model in models:
        model_data = data[data['Model'] == model]
        sdev.append(model_data['Std_Dev_Predicted'].iloc[0]/ref_st)  # 观测标准偏差
        crmsd.append(model_data['RMSE'].iloc[0]/crmsd_max)  # 中心化 RMSD
        ccoef.append(model_data['Correlation'].iloc[0])  # 相关系数

    # 将列表转换为numpy数组
    sdev.insert(0, 1)
    crmsd.insert(0, 0)
    ccoef.insert(0, 1)
    sdev = np.array(sdev)
    crmsd = np.array(crmsd)
    ccoef = np.array(ccoef)
    # 设置绘图参数
    plt.figure(figsize=(8, 6.4))

    # 绘制泰勒图
    sm.taylor_diagram(sdev, crmsd, ccoef, markers=MARKERS, markerLegend='on',labelrms= 'RMSE')

    # 添加标题和标签
    plt.ylabel('Standard deviation')
    plt.savefig(f'E:/GEO/result/qpm/map/{index}taylor.png')

    #plt.show()
'''
生成 Taylor 图

注意，第一个索引对应于图中的参考序列。
例如，sdev[0] 是参考序列的标准偏差，sdev[1:4] 是其他 3 个序列的标准偏差。
sdev[0] 的值用于定义 RMSD 等值线的原点。
其他值用于绘制出现在图表中的点（共 3 个）。

要获取自定义图表的详尽选项列表，请在 Python 命令行中调用该函数：
>> taylor_diagram
'''
