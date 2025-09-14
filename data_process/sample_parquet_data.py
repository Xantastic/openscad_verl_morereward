import pandas as pd
import pyarrow.parquet as pq
import os

# 定义文件路径
base_path = "./"
transfer_data_path = os.path.join(base_path, "transfer_data_verl_compile")

# 获取训练文件和测试文件
train_files = [f for f in os.listdir(transfer_data_path) if f.startswith("train")]
test_files = [f for f in os.listdir(transfer_data_path) if f.startswith("test")]

# 确保有可用的训练文件和测试文件
if not train_files:
    raise FileNotFoundError("未找到训练文件")
if not test_files:
    raise FileNotFoundError("未找到测试文件")

# 选择第一个训练文件和测试文件
train_file_path = os.path.join(transfer_data_path, train_files[0])
test_file_path = os.path.join(transfer_data_path, test_files[0])

print(f"选择的训练文件: {train_files[2]}")
print(f"选择的测试文件: {test_files[0]}")

# 读取训练文件的前200条数据
print("正在读取训练文件的前200条数据...")
train_df = pd.read_parquet(train_file_path).head(200)
print(f"训练数据形状: {train_df.shape}")

# 保存训练数据
train_output_path = os.path.join(base_path, "train_temp_200.parquet")
train_df.to_parquet(train_output_path)
print(f"已保存训练数据到: {train_output_path}")

# 读取测试文件的前800条数据
print("正在读取测试文件的前400条数据...")
test_df = pd.read_parquet(test_file_path).head(100)
print(f"测试数据形状: {test_df.shape}")

# 保存测试数据
test_output_path = os.path.join(base_path, "test_temp_100.parquet")
test_df.to_parquet(test_output_path)
print(f"已保存测试数据到: {test_output_path}")

print("完成!")