import pandas as pd
import json
import numpy as np

# 读取过滤后.parquet文件，查看数据结构，并将第一行数据保存为JSON文件

# 读取parquet文件
file_path = '/home/xa/TempCodes/openscad_verl/transfer_data_verl/train-00001-of-00008.parquet'
df = pd.read_parquet(file_path)

# 显示列名
print("列名:")
for col in df.columns:
    print(f"  {col}")

# 打印数据数量
print("\n数据数量:")
print(len(df))

# 获取第一行数据
one_row = df.iloc[0]

# 处理NumPy数组和标量，转换为Python原生类型
def convert_numpy_types(obj):
    # 处理NumPy数组
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # 处理NumPy标量类型
    elif isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    # 如果是字典，递归处理其值
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    # 如果是列表，递归处理其元素
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

# 将处理后的数据转换为字典
row_dict = {k: convert_numpy_types(v) for k, v in one_row.items()}

# 将第一行保存为当前目录下的JSON文件
with open('one_row_convert.json', 'w', encoding='utf-8') as f:
    json.dump(row_dict, f, ensure_ascii=False, indent=4)

print("已将第一行数据保存到 one_row_convert.json 文件")