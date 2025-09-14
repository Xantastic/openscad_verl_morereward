import pandas as pd
import json
import math
import os


# 将open-r1数据集转换为verl-main所需格式


def process_parquet_files():
    # 从systemprompt.txt文件读取系统提示
    with open('systemprompt.txt', 'r', encoding='utf-8') as file:
        systemprompt = file.read()
    # 读取所有parquet文件
    parquet_files = [
        'org_data/train-00000-of-00008.parquet',
        'org_data/train-00001-of-00008.parquet',
        'org_data/train-00002-of-00008.parquet',
        'org_data/train-00003-of-00008.parquet',
        'org_data/train-00004-of-00008.parquet',
        'org_data/train-00005-of-00008.parquet',
        'org_data/train-00006-of-00008.parquet',
        'org_data/train-00007-of-00008.parquet'
    ]
    


    new_data = []
    global_index = 0
    
    for file_name in parquet_files:
        # 读取parquet文件
        print(f"读取文件: {file_name}")
        df = pd.read_parquet(file_name)
        
        # 处理每一行数据
        for index, row in df.iterrows():
            # 创建新数据结构
            new_row = {}
            
            # 设置数据源
            new_row['data_source'] = 'moneyshredder/openscad'
            
            # 处理prompt字段
            new_row['prompt'] = []
            if 'messages' in row and len(row['messages']) >= 2:
                # 保留user的content作为problem
                user_message = row['messages'][0]
                if user_message['role'] == 'user':
                    # 添加指定的提示内容到用户消息后面
                    content_with_prompt = systemprompt + user_message['content'] 
                    new_row['prompt'].append({
                        'content': content_with_prompt,
                        'role': 'user'
                    })
            
            # 设置ability字段
            new_row['ability'] = 'scad_design'
            
            # 处理reward_model字段
            new_row['reward_model'] = {
                'ground_truth': row['scad_code'] if 'scad_code' in row else '',
                'style': 'rule'  # 保持不变
            }
            
            # 处理extra_info字段
            new_row['extra_info'] = {
                'question': '',
                'answer': '',
                'index': global_index
            }
            
            # 从messages中提取question和answer
            if 'messages' in row and len(row['messages']) >= 2:
                user_message = row['messages'][0]
                assistant_message = row['messages'][1]
                
                if user_message['role'] == 'user':
                    new_row['extra_info']['question'] = user_message['content']
                
                if assistant_message['role'] == 'assistant':
                    new_row['extra_info']['answer'] = assistant_message['content']
            
            # 保留其他没有对应关系的字段
            if 'specificity' in row:
                new_row['specificity'] = row['specificity']

            # 保留其他没有对应关系的字段
            if 'scad_files' in row:
                new_row['scad_files'] = row['scad_files']

            # 添加response字段方便过滤
            if 'scad_code' in row:
                new_row['response'] = row['scad_code']

            global_index += 1
            new_data.append(new_row)
    
    # 创建新的DataFrame
    new_df = pd.DataFrame(new_data)
    total_rows = len(new_df)
    print(f"处理完成，共生成 {total_rows} 行数据")
    
    # 创建输出目录
    output_dir = 'transfer_data_verl'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录: {output_dir}")
    
    # 计算分割点
    # 6份训练数据，1份测试数据，1份验证数据
    train_count = 6
    test_count = 1
    val_count = 1
    
    # 计算每份应该包含的行数
    rows_per_split = math.ceil(total_rows / 8)
    
    # 拆分并保存为多个parquet文件
    for i in range(8):
        # 计算当前分片的起始和结束索引
        start_idx = i * rows_per_split
        end_idx = min((i + 1) * rows_per_split, total_rows)
        
        # 提取当前分片的数据
        split_df = new_df.iloc[start_idx:end_idx]
        
        # 确定文件类型 (train/test/validation)
        if i < 6:
            file_type = 'train'
        elif i == 6:
            file_type = 'test'
        else:
            file_type = 'validation'
        
        # 保存为parquet文件
        filename = f'{output_dir}/{file_type}-0000{str(i)}-of-00008.parquet'
        split_df.to_parquet(filename)
        print(f"已保存 {filename}，包含 {len(split_df)} 行数据")
    
    print("所有数据已拆分为 8 个文件（6份训练，1份测试，1份验证）")


if __name__ == "__main__":
    process_parquet_files()