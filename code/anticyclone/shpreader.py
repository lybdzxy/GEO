import geopandas as gpd
import matplotlib.pyplot as plt

# 替换'path_to_shapefile'为您实际的Shapefile文件路径
shapefile_path = r'E:\GEO\geodata\CNTOTAL.shp'
gdf = gpd.read_file(shapefile_path)

gdf.plot()
plt.show()