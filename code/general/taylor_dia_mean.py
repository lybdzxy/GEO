import pandas as pd

# 读取Excel文件
results_df = pd.read_excel('E:/GEO/etccdi/qpm/model_indices_results_new.xlsx')

# 创建一个Excel写入对象
with pd.ExcelWriter('E:/GEO/etccdi/qpm/model_indices_summary_new.xlsx') as writer:
    # 按照不同指标分组
    for index, group in results_df.groupby('Index'):
        # 在当前工作表中按照模型分类并计算均值
        model_means = group.groupby('Model').mean().reset_index()
        # 写入均值数据到当前工作表
        model_means.to_excel(writer, sheet_name=index, index=False)

# 提示保存完成
print("保存完成。")
