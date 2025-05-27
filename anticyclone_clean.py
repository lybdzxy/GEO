import ee
import pandas as pd

# 初始化 Google Earth Engine
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()  # 首次使用需要认证，请按照提示操作
    ee.Initialize()


def get_elevation(lat, lon):
    """
    使用 COPERNICUS/DEM/GLO30 数据集（通过 ImageCollection.mosaic 合成）获取指定经纬度的海拔（单位：米）。
    若 center_lon 超出 [-180, 180]，则转换到该范围内。
    """
    try:
        # 如果经度大于 180，则转换到 [-180, 180] 范围内
        if lon > 180:
            lon = lon - 360

        # 创建一个点对象（经度, 纬度）
        point = ee.Geometry.Point(lon, lat)
        # 将数据集作为 ImageCollection 载入，并 mosaic 成一幅 Image
        dataset = ee.ImageCollection("COPERNICUS/DEM/GLO30").mosaic()
        # 获取该点的样本数据（scale 设为30米）
        sample = dataset.sample(region=point, scale=30).first()
        if sample is None:
            print(f"警告：未能获取经纬度 ({lat}, {lon}) 的样本数据")
            return None
        # COPERNICUS/DEM/GLO30 数据集的高程信息存储在 "DEM" 波段中
        elevation_value = sample.get("DEM")
        if elevation_value is None:
            print(f"警告：经纬度 ({lat}, {lon}) 的海拔数据为空")
            return None
        return elevation_value.getInfo()
    except Exception as e:
        print(f"无法获取 ({lat}, {lon}) 的海拔信息: {e}")
        return None


def clean_trajectory_data(file_path, output_path):
    """
    读取 CSV 文件，执行以下操作：
      1. 剔除 center_pressure 小于 1020 的所有 trajectory_id 数据；
      2. 通过经纬度获取海拔信息，并过滤掉海拔大于 2000m 或无法获取海拔的记录；
      3. 保存清洗后的数据到新的 CSV 文件。
    """
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 找出 center_pressure 小于 1020 的 trajectory_id
    invalid_ids = df[df['center_pressure'] < 1020]['trajectory_id'].unique()
    # 移除这些 trajectory_id 对应的数据
    df_cleaned = df[~df['trajectory_id'].isin(invalid_ids)]

    '''# 获取海拔信息，并添加到数据中
    df_cleaned['elevation'] = df_cleaned.apply(
        lambda row: get_elevation(row['center_lat'], row['center_lon']), axis=1)

    # 过滤掉海拔信息为 None 或大于 2000m 的数据行
    df_cleaned = df_cleaned[df_cleaned['elevation'].notnull() & (df_cleaned['elevation'] <= 2000)]

    # 如果不需要保留海拔数据，可删除 elevation 列
    df_cleaned.drop(columns=['elevation'], inplace=True)'''

    # 保存清洗后的数据
    df_cleaned.to_csv(output_path, index=False)
    print(f"清洗后的数据已保存至 {output_path}")


if __name__ == '__main__':
    # 修改下面的文件路径以适应你的环境
    input_file = 'trajectories_with_pressure_tot.csv'
    output_file = 'trajectories_with_pressure6024_tot_cleaned.csv'
    clean_trajectory_data(input_file, output_file)
