import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
from matplotlib.colors import BoundaryNorm

indices = ['prcptot', 'r95p', 'r99p', 'sdii', 'rx1day']
ssps = [126, 585]

colors = [
    (103, 0, 31), (178, 24, 43), (214, 96, 77), (244, 165, 130),
    (253, 219, 199), (209, 229, 240), (146, 197, 222), (67, 147, 195),
    (33, 102, 172), (5, 48, 97)
]
scaled_colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]
custom_cmap = ListedColormap(scaled_colors)

china_shapes = list(shpreader.Reader('E:/GEO/geodata/bou2_4l.dbf').geometries())

for ssp in ssps:
        boundary = (-99,-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6,99) if ssp == 126 else (-99,-3.2, -2.4, -1.6, -0.8, 0, 0.8, 1.6, 2.4, 3.2,99)
        norm = BoundaryNorm(boundary, len(custom_cmap.colors))

        # 保存图例
        fig, ax = plt.subplots(figsize=(6, 1))
        fig.subplots_adjust(bottom=0.5)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap),
                            cax=ax, orientation='horizontal')

        # 获取图例刻度
        ticks = [-99,-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6,99] if ssp == 126 else [-99,-3.2, -2.4, -1.6, -0.8, 0, 0.8, 1.6, 2.4, 3.2,99]
        tick_labels = [f'{ticks[i]:.1f}' for i in range(len(ticks))]
        tick_labels[0] = ''
        tick_labels[-1] = ''
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
        plt.savefig(f'E:/GEO/result/new/pic/index_tos_{ssp}_legend.png', dpi=600, bbox_inches='tight')
        plt.close()
