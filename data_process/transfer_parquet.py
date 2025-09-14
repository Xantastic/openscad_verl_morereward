import pandas as pd
import json
import math

def process_parquet_files():
    # 读取所有parquet文件
    parquet_files = [
        '/root/LLM_AD/open-r1/data/openscad_parquet/train-00000-of-00003.parquet',
        '/root/LLM_AD/open-r1/data/openscad_parquet/train-00001-of-00003.parquet',
        '/root/LLM_AD/open-r1/data/openscad_parquet/train-00002-of-00003.parquet'
    ]
    open-r
    new_data = []
    
    for file_name in parquet_files:
        # 读取parquet文件
        df = pd.read_parquet(file_name)
        
        # 处理每一行数据
        for index, row in df.iterrows():
            # 获取cot_responses数据
            cot_responses = row['cot_responses']
            
            # 为每个cot_responses生成一条新数据
            for response in cot_responses:
                # 创建新数据，只保留指定的列
                new_row = {}

                new_row['scad_files'] = row['scad_files']
                
                # 添加cot列
                new_row['scad_cot'] = response['content']
                
                # 添加scad_content列
                new_row['scad_code'] = row['scad_content']
                
                # 添加query列
                new_row['problem'] = response['query']
                
                # 添加specificity列
                new_row['specificity'] = response['specificity']
                
                # 添加reasoning列（如果存在）
                if 'reasoning' in response:
                    new_row['reasoning'] = response['reasoning']
                
                # 添加messages列
                # 使用 <think> 包裹 response['content']
                wrapped_content = f"<think>{response['content']}</think>"
                # assistant的content包括response['content']+row['scad_content']
                answer_content = f"<answer>{row['scad_content']}</answer>"
                assistant_content = wrapped_content + answer_content 

                new_row['solution'] = assistant_content
                
                messages = [
                    {
                        "content": response['query'],
                        "role": "user"
                    },
                    {
                        "content": assistant_content,
                        "role": "assistant"
                    }
                ]
                new_row['messages'] = messages
                
                new_data.append(new_row)
    
    # 创建新的DataFrame
    new_df = pd.DataFrame(new_data)
    total_rows = len(new_df)
    print(f"处理完成，共生成 {total_rows} 行数据")
    
    # 计算每个文件应包含的行数
    num_splits = 8
    rows_per_split = math.ceil(total_rows / num_splits)
    
    # 拆分并保存为多个parquet文件
    for i in range(num_splits):
        # 计算当前分片的起始和结束索引
        start_idx = i * rows_per_split
        end_idx = min((i + 1) * rows_per_split, total_rows)
        
        # 提取当前分片的数据
        split_df = new_df.iloc[start_idx:end_idx]
        
        # 保存为parquet文件
        filename = f'./org_data/train-0000{str(i)}-of-0000{str(num_splits)}.parquet'
        split_df.to_parquet(filename)
        print(f"已保存 {filename}，包含 {len(split_df)} 行数据")
    
    print(f"所有数据已拆分为 {num_splits} 个文件")


if __name__ == "__main__":
    process_parquet_files()
