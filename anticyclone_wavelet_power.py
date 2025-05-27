import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pycwt as wavelet
from scipy import signal
from scipy.interpolate import interp1d

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

for r in range(1, 6):

    # 读取数据
    df = pd.read_csv('annual_region_trajectory_pressure_pivoted.csv')  # 假设数据文件名为 annual_trajectory_amount.csv
    t = df.iloc[:, 0].values  # 时间列（年份，例如1960-2023）
    dat = df.iloc[:, r].values  # 数据列（例如反气旋数量）
    name = df.columns[r]

    # 数据预处理
    dt = 1  # 采样间隔（每年一个数据点）
    dat_detrended = signal.detrend(dat)  # 去除线性趋势
    dat_std = (dat_detrended - np.mean(dat_detrended)) / np.std(dat_detrended)  # 标准化

    # 设置小波分析参数
    mother = wavelet.Morlet(6)  # 使用Morlet小波，中心频率为6
    s0 = 2 * dt  # 最小尺度
    dj = 0.0625  # 尺度分辨率
    J = int(np.log2(len(dat_std) * dt / s0) / dj)  # 尺度数量

    # 进行连续小波变换
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_std, dt, dj, s0, J, mother)

    # 计算小波功率谱
    power = (np.abs(wave)) ** 2

    # 计算显著性和理论噪声谱
    alpha, _, _ = wavelet.ar1(dat_std)  # 使用标准化数据计算红噪声自相关
    N = len(t)  # 时间点数量
    dof = N - scales  # 自由度校正
    signif, fft_theor = wavelet.significance(1, dt, scales, 0, alpha,
                                             significance_level=0.95, dof=dof, wavelet=mother)

    # 计算周期
    period = 1 / freqs

    # 插值 fft_theor 以匹配 period
    log_scales = np.log2(scales)  # 对数尺度
    log_period = np.log2(period)  # 对数周期
    interp_func = interp1d(log_scales, fft_theor, kind='linear', fill_value="extrapolate")
    fft_theor_interp = interp_func(log_period)  # 插值后的理论傅立叶谱

    # 绘图设置：只显示小波功率谱
    plt.close('all')
    plt.ioff()
    figprops = dict(figsize=(3.248, 2), dpi=600)
    fig = plt.figure(**figprops)

    # 调整子图，增加宽度和高度，使其尽量占满整个画布
    ax = plt.axes([0.05, 0.05, 0.9, 0.9])  # 增大子图的尺寸，填满画布

    # 小波功率谱
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    ax.contourf(t, np.log2(period), np.log2(power), np.log2(levels), extend='both', cmap=plt.cm.viridis)
    extent = [t.min(), t.max(), 0, max(period)]
    ax.contour(t, np.log2(period), signif, [-99, 1], colors='k', linewidths=2, extent=extent)
    ax.fill(np.concatenate([t, t[-1:] + dt, t[-1:] + dt, t[:1] - dt, t[:1] - dt]),
            np.concatenate([np.log2(coi), [1e-9], np.log2(period[-1:]), np.log2(period[-1:]), [1e-9]]),
            'k', alpha=0.3, hatch='x')
    ax.set_title(f'{name} 小波功率谱 (Morlet)')
    ax.set_ylabel('周期 (年)')
    Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max()))).astype(int)
    ax.set_yticks(np.log2(Yticks))
    ax.set_yticklabels(Yticks)

    # 保存图形
    # plt.savefig(f'E:/GEO/result/anticyclone/{name}_pressure_wavelet_power_full.png', dpi=600)
    plt.show()
