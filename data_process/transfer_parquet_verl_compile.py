import pandas as pd
import numpy as np
import os
import glob
import threading
import tempfile
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.normalize import normalize_openscad_to_stl
from utils import compute_iou


# 对verl-main数据集进行SCAD编译验证，删除无法编译的行数据（但异步过滤时仍然有问题）

# 全局线程池执行器用于异步处理
compile_executor = None

def verify_scad_compilation(scad_code, use_async=True):
    """
    验证SCAD代码是否能在10秒内编译为STL
    
    Args:
        scad_code (str): SCAD代码
        use_async (bool): 是否使用异步处理模式
        
    Returns:
        bool: 如果能成功编译返回True，否则返回False
    """
    # 创建临时SCAD文件
    temp_scad = None
    result_stl = None
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.scad', delete=False) as temp_scad_file:
            temp_scad = temp_scad_file.name
            temp_scad_file.write(scad_code)
        
        # 使用normalize.py的方法验证编译，启用异步模式
        result_stl = normalize_openscad_to_stl(temp_scad, executor=compile_executor, use_async=use_async)

        iou_value = compute_iou.calculate_3d_iou(result_stl, result_stl) if result_stl else 0.0


        return result_stl is not None and iou_value > 0.99 

        
    except Exception as e:
        print(f"出错: {str(e)}")
        return False
    finally:
        # 清理临时文件
        if temp_scad and os.path.exists(temp_scad):
            try:
                os.remove(temp_scad)
            except OSError:
                pass
        # 清理转换成功的STL文件
        if result_stl and os.path.exists(result_stl):
            try:
                os.remove(result_stl)
            except OSError:
                pass

def process_row(row, row_index, use_async=True):
    """
    处理单行数据，验证SCAD代码是否可编译
    
    Args:
        row: DataFrame行数据
        row_index: 行索引号
        use_async (bool): 是否使用异步处理模式
        
    Returns:
        tuple: (是否保留, 数据字典)
    """
    try:
        # 获取response字段
        if 'response' not in row or pd.isna(row['response']):
            print(f"第{row_index + 1}行数据删除: response字段为空")
            return False, None
            
        scad_code = row['response']
        
        # 验证编译
        if verify_scad_compilation(scad_code, use_async=use_async):
            print(f"第{row_index + 1}行数据保留: SCAD编译成功")
            return True, row.to_dict()
        else:
            print(f"第{row_index + 1}行数据删除: SCAD编译失败")
            return False, None
            
    except Exception as e:
        print(f"第{row_index + 1}行数据删除: 处理异常 - {str(e)}")
        return False, None

# 全局变量用于控制线程池
global_executor = None
compile_executor = None
shutdown_flag = False

def signal_handler(signum, frame):
    """处理程序中断信号"""
    global global_executor, compile_executor, shutdown_flag
    print(f"\n收到中断信号 {signum}，正在优雅关闭...")
    shutdown_flag = True
    if global_executor:
        global_executor.shutdown(wait=False, cancel_futures=True)
    if compile_executor:
        compile_executor.shutdown(wait=False, cancel_futures=True)
    sys.exit(0)

def process_all_data(use_async=True, data_workers=10, compile_workers=100):
    """
    处理所有数据：读取、过滤、拆分保存
    
    Args:
        use_async (bool): 是否使用异步处理模式
        data_workers (int): 数据处理线程池大小
        compile_workers (int): 编译处理线程池大小
    """
    global global_executor, compile_executor
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("开始处理数据...")
    
    # 读取所有parquet文件
    input_dir = '/home/xa/TempCodes/openscad_verl/transfer_data_verl/test-00006-of-00008.parquet'
    parquet_files = glob.glob(input_dir)
    
    if not parquet_files:
        print(f"在 {input_dir} 中未找到parquet文件")
        return
    
    print(f"找到 {len(parquet_files)} 个parquet文件")
    
    # 读取所有数据
    all_data = []
    for file_path in parquet_files:
        print(f"读取文件: {file_path}")
        df = pd.read_parquet(file_path)
        all_data.append(df)
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"合并后共 {len(combined_df)} 行数据")
    
    # 多线程处理数据验证
    async_mode_text = "异步" if use_async else "同步"
    print(f"开始多线程验证SCAD编译（{async_mode_text}模式）...")
    filtered_data = []
    keep_count = 0
    delete_count = 0
    
    try:
        # 创建专用于编译的线程池
        compile_executor = ThreadPoolExecutor(max_workers=compile_workers)
        
        # 创建用于数据处理的线程池
        global_executor = ThreadPoolExecutor(max_workers=data_workers)
        
        # 提交所有任务，传递异步参数
        future_to_row = {global_executor.submit(process_row, row, idx, use_async): (idx, row) 
                        for idx, (_, row) in enumerate(combined_df.iterrows())}
        
        # 收集结果
        completed_count = 0
        shutdown_flag = False
        for future in as_completed(future_to_row):
            if shutdown_flag:
                break
                
            row_idx, original_row = future_to_row[future]
            completed_count += 1
            
            try:
                is_valid, data_dict = future.result()
                
                if is_valid:
                    filtered_data.append(data_dict)
                    keep_count += 1
                else:
                    delete_count += 1
                
                # 每100行显示一次进度
                if completed_count % 100 == 0:
                    print(f"已处理 {completed_count}/{len(combined_df)} 行数据")
                        
            except Exception as e:
                print(f"第{row_idx + 1}行数据处理异常: {str(e)}")
                delete_count += 1
    
    except KeyboardInterrupt:
        print("\n收到键盘中断，正在停止所有线程...")
        shutdown_flag = True
    finally:
        if global_executor:
            global_executor.shutdown(wait=True)
            global_executor = None
        if compile_executor:
            compile_executor.shutdown(wait=True)
            compile_executor = None
    
    print(f"\n验证完成，保留 {len(filtered_data)} 行有效数据")
    
    # 输出处理统计
    print(f"处理统计: 保留 {keep_count} 行，删除 {delete_count} 行")
    
    if not filtered_data:
        print("没有有效数据可保存")
        return
    
    # 创建过滤后的DataFrame
    filtered_df = pd.DataFrame(filtered_data)
    
    # 创建输出目录
    output_dir = './transfer_data_verl_compile'
    os.makedirs(output_dir, exist_ok=True)
    
    # 将数据平均分为6份
    total_rows = len(filtered_df)
    rows_per_split = total_rows // 1
    
    print(f"将数据分为6份，每份约 {rows_per_split} 行")
    
    # 保存分片数据
    for i in range(1):
        start_idx = i * rows_per_split
        if i == 5:  # 最后一份包含剩余所有数据
            end_idx = total_rows
        else:
            end_idx = (i + 1) * rows_per_split
        
        split_df = filtered_df.iloc[start_idx:end_idx]
        
        # 保存文件
        output_file = os.path.join(output_dir, 'temp-test-filter-7.parquet')
        split_df.to_parquet(output_file)
        
        print(f"已保存 {output_file}，包含 {len(split_df)} 行数据")
    
    print("所有数据处理完成！")

if __name__ == "__main__":
    # 使用异步模式处理，可以优化线程池大小
    process_all_data(use_async=True, data_workers=5, compile_workers=10)