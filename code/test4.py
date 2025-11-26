import matplotlib.pyplot as plt
import numpy as np

# ============ 完美解决中文与负号 ============
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============ 你的原始事件（一个都没删） ============
events = [
    ("前寒武纪大冰期",            950_000_000,  600_000_000),
    ("石炭–二叠纪大冰期",         300_000_000,  250_000_000),
    ("第四纪大冰期",               2_600_000,         10_000),
    ("中更新世冰川最盛期 MIS6",     190_000,      130_000),
    ("末次间冰期 MIS5e (Eemian)",   130_000,      115_000),
    ("末次冰期 MIS4",                75_000,       54_000),
    ("末次冰期 MIS3",                54_000,       23_000),
    ("末次冰期 MIS2",                23_000,       10_000),
    ("末次冰期最盛期 LGM",           23_000,       19_000),
    ("新仙女木事件 (Younger Dryas)", 12_900,       11_500),
    ("Heinrich 事件（6次）",         69_000,       14_300),   # 约69–14.3 ka
    ("D–O 快速旋回",         75_000,       15_000),   # 主要发生在MIS3–MIS4
    ("小冰期 LIA (1350–1800 AD)",      700,          150),     # 约650–200 BP，取近似值
]

# ============ 绘图 ============
fig, ax = plt.subplots(figsize=(18, 9))

# 使用对数坐标，但把 0–10 000 年这段单独放大显示
ax.set_xscale('symlog')   # < 10 000 年用线性，> 10 000 年用对数

y = np.arange(len(events))

# 颜色区分：大冰期深蓝，冷事件浅蓝，暖期橙色
colors = ['#084594', '#2171b5', '#4292c6', '#6baed6', '#9ecae1',
          '#c6dbef', '#deebf7', '#fdae6b', '#fd8d3c', '#f16913',
          '#d94801', '#d94801', '#8c2d04']

for i, (name, start, end) in enumerate(events):
    ax.barh(y[i], start - end, left=end, height=0.65, color=colors[i], edgecolor='black', linewidth=0.8)
    # 文字写在条形右侧（现代方向）
    ax.text(start + (1e9 if start > 1e8 else 5e5 if start > 1e6 else 800),
            y[i], name, va='center', ha='left', fontsize=13, fontweight='bold')

# ============ 美化 ============
ax.set_ylim(-1, len(events))
ax.set_yticks([])
ax.set_xlabel('距今时间（年 BP）', fontsize=16)
ax.invert_xaxis()   # 让时间从右（过去）到左（现在），符合日常阅读习惯

# 添加几条关键竖线
ax.axvline(11_700, color='green', lw=2, alpha=0.8, label='全新世开始')
ax.axvline(2_600_000, color='red', lw=2, alpha=0.7, label='第四纪开始')
ax.legend(loc='lower right', fontsize=12)

ax.set_title('《冰冻圈科学》4.1 冰冻圈的过去 —— 主要冰期时间轴\n'
             '吴逸潇 202521051015',
             fontsize=20, pad=20, loc='center')

ax.grid(True, axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()