import subprocess
import os
import trimesh
import re
import tempfile
import gc
def normalize_openscad_to_stl(scad_file, openscad_path="/usr/bin/openscad-nightly"):
    """
    将OpenSCAD文件经过normalize处理生成STL文件
    
    Args:
        scad_file (str): 输入的OpenSCAD文件路径
        openscad_path (str): OpenSCAD可执行文件路径
        
    Returns:
        str: 生成的归一化STL文件路径，如果失败则返回None
    """
    try:
        # 使用normalize.py中的方法处理OpenSCAD模型
        output_stl, normalized_scad = process_openscad_model(scad_file, openscad_path=openscad_path)
        
        # 清理临时的SCAD文件
        if os.path.exists(normalized_scad):
            os.remove(normalized_scad)
            
        return output_stl
    except Exception as e:
        # print(f"处理OpenSCAD文件时出错: {str(e)}")
        return None


# def extract_answer_content_regex(content):
#     match = re.search(r'</think>\s*<answer>(.*?)</answer>', content, re.DOTALL | re.IGNORECASE)
#     return match.group(1).strip() if match else ""


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def count_answer_tags(text):

    opening_tags_answer = text.count("<answer>")
    closing_tags_answer = text.count("</answer>")

    opening_tags_think = text.count("<think>")
    closing_tags_think = text.count("</think>")
    
    return opening_tags_answer, closing_tags_answer, opening_tags_think, closing_tags_think


def convert_openscad_to_stl(scad_file, stl_file, openscad_path="/usr/bin/openscad-nightly"):
    """将OpenSCAD文件转换为STL格式"""
    try:
        subprocess.run(
            [openscad_path, "-o", stl_file,"--backend", "Manifold", scad_file],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10
        )
        # print(f"成功将 {scad_file} 转换为 {stl_file}")
        return True
    except subprocess.CalledProcessError as e:
        # print(f"转换失败: {e.stderr}")
        return False
    except FileNotFoundError:
        # print(f"未找到OpenSCAD可执行文件，请检查路径是否正确: {openscad_path}")
        return False
    except subprocess.TimeoutExpired:
        # print(f"转换超时: {scad_file}")
        return False


def get_normalization_parameters(input_stl):
    """获取模型的归一化参数：平移向量和缩放因子"""
    # 加载STL模型
    mesh = trimesh.load(input_stl)
    
    try:
        # 获取边界框 (min, max)
        bounds = mesh.bounds
        min_coords, max_coords = bounds

        # 计算中心点
        center = (min_coords + max_coords) / 2

        # 计算各维度尺寸和最大尺寸
        sizes = max_coords - min_coords
        max_size = max(sizes)
        scale_factor = 1.0 / max_size if max_size > 0 else 0

        return center, scale_factor, bounds, sizes
    finally:
        # Ensure mesh is properly closed to free memory
        if hasattr(mesh, 'close'):
            mesh.close()
        del mesh
        gc.collect()


def create_normalized_scad(original_scad, center):
    """创建包含归一化处理代码的新SCAD文件，不修改原文件（仅平移，不缩放）"""
    # 生成新文件名和归一化模块的名称
    base_name = os.path.splitext(os.path.basename(original_scad))[0]
    normalized_scad = f"{os.path.splitext(original_scad)[0]}_normalized.scad"
    normalized_module = f"{base_name}_normalized"

    # 读取原始文件内容
    with open(original_scad, 'r', encoding='utf-8') as f:
        original_content = f.read()

    # 检查原始文件是否已经包含主模块定义
    if f"module {base_name}()" not in original_content:
        # 如果没有，将原始内容包装到一个模块中
        wrapped_content = f"module {base_name}() {{\n"
        wrapped_content += original_content
        wrapped_content += "\n}\n"
    else:
        # 直接使用原始内容
        wrapped_content = original_content

    # 准备要添加的归一化代码（仅平移，不缩放）
    normalization_code = f"""

// 以下是自动添加的归一化处理代码
module {normalized_module}() {{
    // 归一化参数
    // 原始模型中心点: {center.tolist()}

    // 仅平移到原点，不进行缩放（保持原始尺寸）
    translate({(-center).tolist()}) {{
        // 引用原始模型
        {base_name}();
    }}
}}

// 调用归一化模块以显示结果
{normalized_module}();
"""

    # 写入新的SCAD文件
    with open(normalized_scad, 'w', encoding='utf-8') as f:
        f.write(wrapped_content)
        f.write(normalization_code)

    # print(f"已创建包含归一化代码的新SCAD文件: {normalized_scad}")
    return normalized_scad, normalized_module


def process_openscad_model(scad_file, output_stl=None, openscad_path="/usr/bin/openscad-nightly"):
    """
    完整处理流程：
    1. 将原始SCAD转换为临时STL以获取参数
    2. 计算归一化参数（仅平移，不缩放）
    3. 创建包含归一化代码的新SCAD文件（不修改原文件）
    4. 使用新的SCAD文件导出归一化STL
    """
    # 生成临时文件名 with proper cleanup
    base_name = os.path.splitext(os.path.basename(scad_file))[0]
    temp_stl = None
    temp_stl_obj = None
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.stl', delete=False) as temp_stl_obj:
            temp_stl = temp_stl_obj.name

        # 如果未指定输出文件名，则自动生成
        if output_stl is None:
            output_stl = f"{base_name}_normalized.stl"

        # 第一步：转换原始SCAD到临时STL以获取参数
        if not convert_openscad_to_stl(scad_file, temp_stl, openscad_path):
            return None

        # 第二步：获取归一化参数
        center, scale_factor, bounds, sizes = get_normalization_parameters(temp_stl)

        # 输出处理信息
        # print(f"原始边界框: {bounds}")
        # print(f"原始中心点: {center.tolist()}")
        # print(f"原始尺寸: {sizes.tolist()}")
        # print(f"保持原始尺寸（不缩放），仅平移到原点")

        # 第三步：创建包含归一化代码的新SCAD文件（仅平移，不缩放）
        normalized_scad, normalized_module = create_normalized_scad(scad_file, center)

        # 第四步：使用新的SCAD文件导出归一化STL
        # print(f"从新的SCAD文件导出归一化STL...")
        success = convert_openscad_to_stl(normalized_scad, output_stl, openscad_path)

        if success:
            # print(f"已成功导出归一化STL文件: {output_stl}")
            return output_stl, normalized_scad
        else:
            # print("导出归一化STL失败")
            return None, normalized_scad
            
    except Exception as e:
        # print(f"处理OpenSCAD模型时出错: {str(e)}")
        return None, None
    finally:
        # 清理临时文件
        if temp_stl and os.path.exists(temp_stl):
            try:
                os.remove(temp_stl)
            except (OSError, PermissionError):
                pass
        
        # Force garbage collection
        gc.collect()


# if __name__ == "__main__":
#     # 示例用法
#     input_scad = "C:\\Users\\13093\\Desktop\\ICLR\\Test_Openscad\\2226_l_bracket.scad"  # 替换为你的OpenSCAD文件
#     output_stl = "normalized_model_bracket.stl"  # 输出的归一化STL模型

#     # OpenSCAD的安装路径
#     openscad_path = "E:\\OpenSCAD\\openscad.exe"

#     # 处理模型
#     process_openscad_model(input_scad, output_stl, openscad_path)
