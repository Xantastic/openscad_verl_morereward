import json
import os


# 验证数据中的SCAD代码保存为scad文件后，是否可以在Openscad软件正确渲染

def save_scad_from_json(json_file_path, output_file_name):
    """
    从 JSON 文件中读取 reward_model 的 ground_truth 并保存为 .scad 文件
    
    Args:
        json_file_path (str): JSON 文件路径
        output_file_name (str): 输出的 .scad 文件名
    """
    try:
        # 读取 JSON 文件
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # 提取 reward_model 中的 ground_truth
        if 'reward_model' in data and 'ground_truth' in data['reward_model']:
            scad_code = data['reward_model']['ground_truth']
            
            # 保存为 .scad 文件
            output_path = os.path.join(os.path.dirname(json_file_path), output_file_name)
            with open(output_path, 'w', encoding='utf-8') as scad_file:
                scad_file.write(scad_code)
            
            print(f"成功保存 SCAD 文件: {output_path}")
            return output_path
        else:
            print("错误: JSON 文件中未找到 reward_model.ground_truth")
            return None
            
    except FileNotFoundError:
        print(f"错误: 找不到文件 {json_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"错误: {json_file_path} 不是有效的 JSON 文件")
        return None
    except Exception as e:
        print(f"错误: {str(e)}")
        return None

if __name__ == "__main__":
    # 设置文件路径
    json_file_path = "one_row_temp.json"
    output_file_name = "ground_truth.scad"
    
    # 执行保存操作
    saved_file = save_scad_from_json(json_file_path, output_file_name)
    
    if saved_file:
        # 显示文件的前几行以验证
        print("\n生成的 SCAD 文件前 10 行:")
        with open(saved_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                print(f"{i+1}: {line.rstrip()}")

