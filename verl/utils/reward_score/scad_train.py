import os
import tempfile
import trimesh
import gc
import logging

# Import the required functions from existing modules
from .ioutils.iou_func import *
save_path ="/home/yc27979/xa/openscad_verl_newcompute/temp_stl"

from transformers import AutoTokenizer

def load_tokenizer(tokenizer_dir="/home/yc27979/xa/openscad_verl_newcompute_onlygrpo_minpunishall_qwen3/verl/utils/reward_score/ioutils"):
    """Load tokenizer from directory"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None
tokenizer = load_tokenizer()

def get_mesh_scale(mesh):
    """
    计算网格模型的尺寸大小，返回最大维度值
    
    Args:
        mesh: trimesh.Trimesh对象
        
    Returns:
        float: 模型的最大维度（宽度、高度或深度）
    """
    if mesh is None:
        return 1.0
    
    # 获取模型的边界框
    bounds = mesh.bounds
    if bounds is None:
        return 1.0
    
    # 计算各维度尺寸
    dimensions = bounds[1] - bounds[0]  # max - min
    
    # 返回最大维度作为模型尺寸
    return max(dimensions)

def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0, max_length=None, token_length=None):
    """
    Compute score for SCAD solutions with proper temporary file management.
    
    Args:
        solution_str: The solution string containing SCAD code
        ground_truth: The ground truth SCAD code
        method: Scoring method (unused in this implementation)
        format_score: Score for format correctness
        score: Maximum score for correct solution
        max_length: Length of the longest answer in the batch (for length-based adjustments)
        
    Returns:
        float: Computed score between 0.0 and score
    """
    content = solution_str
    # sol = ground_truth
    sol_cot = extract_cot(ground_truth)
    sol = extract_solution(ground_truth)
    
    open_count, close_count, open_count_think, close_count_think = count_answer_tags(solution_str)
    
    content_cot = extract_cot(content)
    content = extract_solution(content)
    
    
    
    if content is None:
        return 0.0
    
    # Initialize variables for resource tracking
    content_file_obj = None
    sol_file_obj = None
    content_file_path = None
    sol_file_path = None
    temp_stl1 = None
    temp_stl2 = None
    mesh1 = None
    mesh2 = None
    mesh1_normalized = None
    mesh2_normalized = None
    
    try:
        # Create temporary files with proper error handling
        content_file_obj = tempfile.NamedTemporaryFile(mode='w', suffix='.scad', delete=False, dir=save_path)
        content_file_path = content_file_obj.name
        content_file_obj.write(content)
        content_file_obj.close()  # Close the file so OpenSCAD can access it

        # Convert to STL using the new processing method
        temp_stl1_obj = tempfile.NamedTemporaryFile(mode='w', suffix='.stl', delete=False, dir=save_path)
        temp_stl1 = temp_stl1_obj.name
        temp_stl1_obj.close()
        
        success1 = convert_openscad_to_stl(content_file_path, temp_stl1)
        if not success1:
            return 0.0
        
        # Load STL file and normalize center
        mesh1 = trimesh.load(temp_stl1)
        mesh1_normalized = center_mesh(mesh1)
        
        # Clean up temporary STL file
        if os.path.exists(temp_stl1):
            os.remove(temp_stl1)
            temp_stl1 = None
        
        sol_file_obj = tempfile.NamedTemporaryFile(mode='w', suffix='.scad', delete=False, dir=save_path)
        sol_file_path = sol_file_obj.name
        sol_file_obj.write(sol)
        sol_file_obj.close()  # Close the file so OpenSCAD can access it
            
        # Convert ground truth to STL using the new processing method
        temp_stl2_obj = tempfile.NamedTemporaryFile(mode='w', suffix='.stl', delete=False, dir=save_path)
        temp_stl2 = temp_stl2_obj.name
        temp_stl2_obj.close()
        
        success2 = convert_openscad_to_stl(sol_file_path, temp_stl2)
        if not success2:
            return 0.0
        
        # Load STL file and normalize center
        mesh2 = trimesh.load(temp_stl2)
        mesh2_normalized = center_mesh(mesh2)
        
        # Clean up temporary STL file
        if os.path.exists(temp_stl2):
            os.remove(temp_stl2)
            temp_stl2 = None
        
        # Calculate IOU score directly using mesh objects
        iou_value = calculate_3d_iou(mesh1_normalized, mesh2_normalized)
        # Calculate Chamfer Distance and similarity
        chamfer_dist = chamfer_distance(mesh1_normalized, mesh2_normalized)
        model_scale = get_mesh_scale(mesh2)
        chamfer_value = model_scale / (model_scale + chamfer_dist)  # Dynamic scaling based on model size
        # Calculate volume similarity
        volume_value = volume_similarity(mesh1_normalized, mesh2_normalized)
        # Calculate Surface Similarity
        surface_value = surface_area_similarity(mesh1_normalized, mesh2_normalized)
        # Calculate Bounding Box Overlap Similarity
        bbox_value = bbox_overlap_similarity(mesh1_normalized, mesh2_normalized)

        reward_final = iou_value * 3 +  volume_value +  surface_value + bbox_value
        # print("------------------------------")
        # print(f"IOU: {iou_value:.4f}, Chamfer Sim: {chamfer_value:.4f}, Volume Sim: {volume_value:.4f}, Surface Sim: {surface_value:.4f}, Total Reward: {reward_final:.4f}")
        
        # *********************Scale to code part*********************
        res_tokens_code = tokenizer.encode(content, truncation=True, max_length=1000000)
        res_tokens_code_length = len(res_tokens_code)

        gt_tokens_code = tokenizer.encode(sol, truncation=True, max_length=1000000)
        gt_tokens_code_length = len(gt_tokens_code)

        if res_tokens_code_length < gt_tokens_code_length:
            scale_code = res_tokens_code_length / gt_tokens_code_length
        else:
            scale_code = gt_tokens_code_length / res_tokens_code_length

        # scale_code = res_tokens_code_length / gt_tokens_code_length
        scale_code = min(scale_code, 1.0)
        scale_code = max(scale_code, 0.5)
        # print("scale_code:" + str(scale_code))
        
        # ********************Scale to cot part*********************
        res_tokens_cot = tokenizer.encode(content_cot, truncation=True, max_length=1000000)
        res_tokens_cot_length = len(res_tokens_cot)

        gt_tokens_cot = tokenizer.encode(sol_cot, truncation=True, max_length=1000000)
        gt_tokens_cot_length = len(gt_tokens_cot)
        
        if res_tokens_cot_length < gt_tokens_cot_length:
            scale_cot = res_tokens_cot_length / gt_tokens_cot_length
        else:
            scale_cot = gt_tokens_cot_length / res_tokens_cot_length

        # scale_cot = res_tokens_cot_length / gt_tokens_cot_length
        scale_cot = min(scale_cot, 1.0)
        scale_cot = max(scale_cot, 0.5)
        # print("scale_cot:" + str(scale_cot))
        
        scale = min(scale_code,scale_cot)


        # *********************Scale to All answer*********************
        # res_tokens = tokenizer.encode(solution_str, truncation=True, max_length=1000000)
        # res_tokens_length = len(res_tokens)

        # gt_tokens = tokenizer.encode(ground_truth, truncation=True, max_length=1000000)
        # gt_tokens_length = len(gt_tokens)

        # if res_tokens_length > gt_tokens_length:
        #     scale = gt_tokens_length / res_tokens_length
        #     scale = min(scale, 1.0)
        #     scale = max(scale, 0.8)
        #     print("res > gt scale:" + str(scale))
        # else:
        #     print("gt < res  scale:" + str(scale))
            
            
        if scale > 0.8:
            scale = 1.0

        # Apply penalties for excessive answer tags
        if open_count > 5 or close_count > 5 or open_count_think > 5 or close_count_think > 5 or res_tokens_cot_length == 0 or gt_tokens_cot_length == 0:
            reward_final = 0.0
        
        # Apply length-based reward adjustments if max_length is provided
        # if max_length is not None and token_length is not None:

        #     iou_value = apply_length_based_reward_adjustment(iou_value, token_length, max_length)
        
        # print("scale:" + str(scale))
            
        return reward_final * scale
        
    except Exception as e:
        # Log the error for debugging (consider using proper logging in production)
        logging.warning(f"Error in SCAD score computation: {e}")
        return 0.0
    finally:
        # Ensure all temporary files are cleaned up properly
        temp_files_to_remove = [
            content_file_path, 
            sol_file_path, 
            temp_stl1, 
            temp_stl2
        ]
        
        for temp_file in temp_files_to_remove:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except (OSError, PermissionError) as cleanup_error:
                    # Log cleanup errors but don't fail the function
                    logging.warning(f"Failed to remove temporary file {temp_file}: {cleanup_error}")
        
        # Ensure file objects are properly closed
        for file_obj in [content_file_obj, sol_file_obj]:
            if file_obj and not file_obj.closed:
                try:
                    file_obj.close()
                except:
                    pass
        
        # Clean up mesh objects
        for mesh in [mesh1, mesh2, mesh1_normalized, mesh2_normalized]:
            if mesh is not None:
                if hasattr(mesh, 'close'):
                    try:
                        mesh.close()
                    except:
                        pass
                del mesh
        
        # Force garbage collection
        gc.collect()


def process_scad_files_to_iou(scad_file1, scad_file2, openscad_path="/usr/bin/openscad-nightly"):
    """
    处理两个SCAD文件，转换为STL，进行中心归一化，然后计算IOU
    
    Args:
        scad_file1 (str): 第一个SCAD文件路径
        scad_file2 (str): 第二个SCAD文件路径  
        openscad_path (str): OpenSCAD可执行文件路径
        
    Returns:
        tuple: (iou_value, None, None) - IOU值（不再返回STL文件路径）
    """
    # 创建临时文件用于存储STL
    temp_stl1 = None
    temp_stl2 = None
    temp_stl1_obj = None
    temp_stl2_obj = None
    mesh1 = None
    mesh2 = None
    mesh1_normalized = None
    mesh2_normalized = None
    
    try:
        # 创建临时STL文件
        temp_stl1_obj = tempfile.NamedTemporaryFile(mode='w', suffix='.stl', delete=False, dir=save_path)
        temp_stl1 = temp_stl1_obj.name
        temp_stl1_obj.close()

        temp_stl2_obj = tempfile.NamedTemporaryFile(mode='w', suffix='.stl', delete=False, dir=save_path)
        temp_stl2 = temp_stl2_obj.name
        temp_stl2_obj.close()
        
        # print(f"转换 {scad_file1} 到 STL...")
        # 转换第一个SCAD文件到STL
        success1 = convert_openscad_to_stl(scad_file1, temp_stl1, openscad_path)
        if not success1:
            print(f"转换 {scad_file1} 失败")
            return 0.0, None, None
            
        # print(f"转换 {scad_file2} 到 STL...")
        # 转换第二个SCAD文件到STL
        success2 = convert_openscad_to_stl(scad_file2, temp_stl2, openscad_path)
        if not success2:
            print(f"转换 {scad_file2} 失败")
            return 0.0, None, None
        
        # print(f"加载STL文件并进行中心归一化...")
        # 加载STL文件并转换为mesh
        mesh1 = trimesh.load(temp_stl1)
        mesh2 = trimesh.load(temp_stl2)
        
        # 进行中心归一化
        mesh1_normalized = center_mesh(mesh1)
        mesh2_normalized = center_mesh(mesh2)
        
        # 清理临时STL文件
        if os.path.exists(temp_stl1):
            os.remove(temp_stl1)
            temp_stl1 = None
        if os.path.exists(temp_stl2):
            os.remove(temp_stl2)
            temp_stl2 = None
        
        print(f"计算3D IOU...")
        # 直接使用mesh对象计算IOU
        iou_value = calculate_3d_iou(mesh1_normalized, mesh2_normalized)
        
        print(f"IOU计算完成: {iou_value:.4f}")
        
        return iou_value, None, None
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        return 0.0, None, None
        
    finally:
        # 清理临时文件和内存
        temp_files = [temp_stl1, temp_stl2]
        
        # 清理临时STL文件
        for temp_file in temp_files:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except (OSError, PermissionError):
                    pass
        
        # 确保文件对象被正确关闭
        for file_obj in [temp_stl1_obj, temp_stl2_obj]:
            if file_obj and not file_obj.closed:
                try:
                    file_obj.close()
                except:
                    pass
        
        # 清理mesh对象
        for mesh in [mesh1, mesh2, mesh1_normalized, mesh2_normalized]:
            if mesh is not None:
                if hasattr(mesh, 'close'):
                    try:
                        mesh.close()
                    except:
                        pass
                del mesh
        
        # 强制垃圾回收
        gc.collect()


def main():
    """
    主函数：使用test.scad和test1.scad进行测试
    """
    # 测试文件路径
    scad_file1 = "test.scad"
    scad_file2 = "test1.scad"
    
    # 检查文件是否存在
    if not os.path.exists(scad_file1):
        print(f"错误: 找不到文件 {scad_file1}")
        return
        
    if not os.path.exists(scad_file2):
        print(f"错误: 找不到文件 {scad_file2}")
        return
    
    print("开始处理SCAD文件...")
    print(f"文件1: {scad_file1}")
    print(f"文件2: {scad_file2}")
    
    # 处理文件并计算IOU
    iou_value, stl_file1, stl_file2 = process_scad_files_to_iou(scad_file1, scad_file2)
    
    if iou_value > 0:
        print(f"\n最终结果:")
        print(f"IOU值: {iou_value:.4f}")
        print(f"生成的STL文件:")
        if stl_file1:
            print(f"  - {stl_file1}")
        if stl_file2:
            print(f"  - {stl_file2}")
    else:
        print("\n处理失败，请检查OpenSCAD安装路径和SCAD文件内容")


if __name__ == "__main__":
    main()