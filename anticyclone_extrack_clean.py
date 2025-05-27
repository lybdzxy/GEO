import pandas as pd
import re


def filter_rows(input_file, output_file, date_column='date'):
    """
    读取CSV文件并:
    1. 剔除date列格式不符合YYYYMMDDHH的行
    2. 剔除包含混合数据类型的行
    3. 最后只保留前六列数据

    参数:
        input_file (str): 输入CSV文件路径
        output_file (str): 输出CSV文件路径
        date_column (str): 包含日期的列名，默认为'date'
    """
    # 定义日期格式的正则表达式 (YYYYMMDDHH)
    date_pattern = re.compile(r'^\d{10}$')

    try:
        # 第一次读取：检测混合类型列
        print("正在检测混合数据类型列...")
        df_dtypes = pd.read_csv(input_file, nrows=594517)  # 只读取前100行用于检测类型
        consistent_cols = []

        for col in df_dtypes.columns:
            # 检查列是否只有一种数据类型
            if len(df_dtypes[col].apply(type).unique()) == 1:
                consistent_cols.append(col)
            else:
                print(f"发现混合数据类型列: {col} - 将剔除包含此列非一致类型的行")

        # 第二次读取：完整读取并过滤
        print("正在读取并过滤数据...")
        df = pd.read_csv(input_file, low_memory=False)

        # 检查date列是否存在
        if date_column not in df.columns:
            raise ValueError(f"列 '{date_column}' 不存在于CSV文件中")

        # 将date列转换为字符串类型以确保匹配
        df[date_column] = df[date_column].astype(str)

        # 过滤1：只保留符合YYYYMMDDHH格式的日期
        date_filter = df[date_column].str.match(date_pattern)

        # 过滤2：剔除混合类型行
        type_filter = pd.Series(True, index=df.index)
        for col in df.columns:
            if col not in consistent_cols:
                # 对于混合类型列，检查每行是否与第一行的类型一致
                ref_type = type(df[col].iloc[0])
                type_filter &= df[col].apply(lambda x: isinstance(x, ref_type))

        # 应用双重过滤
        filtered_df = df[date_filter & type_filter]

        # 只保留前六列
        filtered_df = filtered_df.iloc[:, :6]

        # 保存到新文件
        filtered_df.to_csv(output_file, index=False)

        print("\n处理结果统计:")
        print(f"原始行数: {len(df)}")
        print(f"因日期格式不符删除的行数: {len(df) - len(df[date_filter])}")
        print(f"因混合数据类型删除的行数: {len(df[date_filter]) - len(filtered_df)}")
        print(f"最终保留行数: {len(filtered_df)}")
        print(f"结果已保存到: {output_file}")

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")


# 使用示例
if __name__ == "__main__":
    input_csv = "high_pressure_systems6024.csv"  # 替换为你的输入文件路径
    output_csv = "output_filtered.csv"  # 替换为你的输出文件路径

    filter_rows(input_csv, output_csv)
