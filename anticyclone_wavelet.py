import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pycwt as wavelet
from scipy import signal
from scipy.interpolate import interp1d

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
df = pd.read_csv('annual_trajectory_pressure.csv')  # 假设数据文件名为 annual_trajectory_amount.csv
t = df.iloc[:, 0].values  # 时间列（年份，例如1960-2023）
dat = df.iloc[:, 1].values  # 数据列（例如反气旋数量）

# 2. 数据预处理
dt = 1  # 采样间隔（每年一个数据点）
dat_detrended = signal.detrend(dat)  # 去除线性趋势
dat_std = (dat_detrended - np.mean(dat_detrended)) / np.std(dat_detrended)  # 标准化

# 3. 设置小波分析参数
mother = wavelet.Morlet(6)  # 使用Morlet小波，中心频率为6
s0 = 2 * dt  # 最小尺度
dj = 0.0625  # 尺度分辨率
J = int(np.log2(len(dat_std) * dt / s0) / dj)  # 尺度数量

# 4. 进行连续小波变换
wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_std, dt, dj, s0, J, mother)

# 5. 计算小波功率谱
power = (np.abs(wave)) ** 2

# 6. 计算逆小波变换
iwave = wavelet.icwt(wave, scales, dt, dj, mother).real

# 7. 计算全局小波谱
glbl_power = np.mean(power, axis=1)

# 8. 计算傅立叶谱
fft_power = np.abs(fft) ** 2

# 9. 计算显著性和理论噪声谱
alpha, _, _ = wavelet.ar1(dat_std)  # 使用标准化数据计算红噪声自相关
N = len(t)  # 时间点数量
dof = N - scales  # 自由度校正
signif, fft_theor = wavelet.significance(1, dt, scales, 0, alpha,
                                         significance_level=0.95, dof=dof, wavelet=mother)

# 10. 计算周期
period = 1 / freqs

# 11. 插值 fft_theor 以匹配 period
log_scales = np.log2(scales)  # 对数尺度
log_period = np.log2(period)  # 对数周期
interp_func = interp1d(log_scales, fft_theor, kind='linear', fill_value="extrapolate")
fft_theor_interp = interp_func(log_period)  # 插值后的理论傅立叶谱

# 12. 计算尺度平均小波谱（2-8年）
sel = np.where((period >= 2) & (period < 8))[0]  # 找到 2-8 年的索引
Cdelta = mother.cdelta  # 小波常数
scale_avg = (scales * np.ones((N, 1))).transpose()  # 扩展尺度
scale_avg = power / scale_avg  # 按 Torrence 和 Compo (1998) 公式 24 归一化
scale_avg = np.var(dat_std) * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)  # 尺度平均功率
scale_avg_signif, tmp = wavelet.significance(np.var(dat_std), dt, scales, 2, alpha,
                                             significance_level=0.95,
                                             dof=[scales[sel[0]],
                                                  scales[sel[-1]]],
                                             wavelet=mother)

# 13. 显著性水平（简化处理）
signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                         significance_level=0.95,
                                         wavelet=mother)
sig951 = np.ones([1, N]) * signif[:, None]
sig95 = power / sig951
glbl_power = power.mean(axis=1)
dof = N - scales  # Correction for padding at edges
glbl_signif, tmp = wavelet.significance(np.var(dat_std), dt, scales, 1, alpha,
                                        significance_level=0.95, dof=dof,
                                        wavelet=mother)


# 14. 绘图设置
plt.close('all')
plt.ioff()
figprops = dict(figsize=(11, 8), dpi=200)
fig = plt.figure(**figprops)
fig.suptitle('北半球温带反气旋平均中心气压小波分析')

# 子图1：原始数据和逆小波变换
ax = plt.axes([0.1, 0.75, 0.65, 0.15])
ax.plot(t, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5], label='逆小波变换')
ax.plot(t, dat_std, 'k', linewidth=1.5, label='标准化数据')
ax.set_title('a) 标准化数据和逆小波变换')
ax.set_ylabel('幅度')
ax.legend()

# 子图2：小波功率谱
bx = plt.axes([0.1, 0.37, 0.65, 0.28], sharex=ax)
levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
bx.contourf(t, np.log2(period), np.log2(power), np.log2(levels), extend='both', cmap=plt.cm.viridis)
extent = [t.min(), t.max(), 0, max(period)]
bx.contour(t, np.log2(period), sig95, [-99, 1], colors='k', linewidths=2, extent=extent)
bx.fill(np.concatenate([t, t[-1:] + dt, t[-1:] + dt, t[:1] - dt, t[:1] - dt]),
        np.concatenate([np.log2(coi), [1e-9], np.log2(period[-1:]), np.log2(period[-1:]), [1e-9]]),
        'k', alpha=0.3, hatch='x')
bx.set_title('b) 小波功率谱 (Morlet)')
bx.set_ylabel('周期 (年)')
Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
bx.set_yticks(np.log2(Yticks))
bx.set_yticklabels(Yticks)

# 子图3：全局小波和傅立叶谱
cx = plt.axes([0.77, 0.37, 0.2, 0.28], sharey=bx)
cx.plot(glbl_signif, np.log2(period), 'k--', label='显著性')
cx.plot(fft_theor_interp, np.log2(period), '--', color='#cccccc', label='理论傅立叶')
cx.plot(fft_power, np.log2(1./fftfreqs), '-', color='#cccccc', linewidth=1., label='傅立叶功率')
cx.plot(glbl_power, np.log2(period), 'k-', linewidth=1.5, label='全局小波')
cx.set_title('c) 全局小波谱')
cx.set_xlabel('功率')
cx.set_xlim([0, glbl_power.max() + 1])
cx.set_ylim(np.log2([period.min(), period.max()]))
cx.set_yticks(np.log2(Yticks))
cx.set_yticklabels(Yticks)
plt.setp(cx.get_yticklabels(), visible=False)
cx.legend()

# 子图4：尺度平均小波谱
dx = plt.axes([0.1, 0.07, 0.65, 0.2], sharex=ax)
dx.axhline(scale_avg_signif, color='k', linestyle='--', linewidth=1., label='显著性')
dx.plot(t, scale_avg, 'k-', linewidth=1.5, label='尺度平均功率')
dx.set_title('d) 2-8 年尺度平均功率')
dx.set_xlabel('时间 (年)')
dx.set_ylabel('平均方差')
ax.set_xlim([t.min(), t.max()])
dx.legend()

# 显示图形
plt.show()
