import trimesh
import numpy as np
import gc



def calculate_single_iou(mesh1, mesh2):
    """
    计算两个mesh的IOU
    """
    # 检查模型是否为水密（表面闭合）
    if not mesh1.is_watertight:
        raise ValueError("模型1不是水密的，请修复STL文件")
    if not mesh2.is_watertight:
        raise ValueError("模型2不是水密的，请修复STL文件")

    # 计算单个模型的体积
    vol1 = mesh1.volume
    vol2 = mesh2.volume

    # 计算交集体积
    try:
        # 计算两个模型的交集（返回新的网格）
        intersection = mesh1.intersection(mesh2)
        vol_intersect = intersection.volume if intersection else 0.0
    except Exception as e:
        # 若交集计算失败（无重叠或模型复杂），视为0
        vol_intersect = 0.0

    # 计算并集体积
    vol_union = vol1 + vol2 - vol_intersect

    # 计算IOU
    if vol_union < 1e-9:  # 避免除以0
        return 0.0
    return vol_intersect / vol_union


def calculate_3d_iou(stl_path1, stl_path2):
    """
    计算两个STL模型的3D IOU（交并比）
    """
    # 1. 读取STL文件
    mesh1 = trimesh.load(stl_path1)
    mesh2 = trimesh.load(stl_path2)
    
    try:

        try:
            iou_x = calculate_single_iou(mesh1, mesh2)
            # print(iou_x)
            return iou_x

        except Exception as e:
            print(f"计算IOU时出错: {str(e)}")
            return 0.0
                
    finally:
        # Clean up mesh objects to prevent memory leaks
        for mesh in [mesh1, mesh2]:
            if hasattr(mesh, 'close'):
                mesh.close()
            del mesh
        
        gc.collect()

# # ---------------------- 测试代码 ----------------------
# if __name__ == "__main__":
#     # 替换为你的两个 STL 文件路径
#     stl1_path = "normalized_model.stl"
#     stl2_path = "output.stl"

#     try:
#         iou_value = calculate_3d_iou(stl1_path, stl2_path)
#         print(f"两个 3D 模型的 IOU：{iou_value:.4f}")
#     except Exception as e:
#         print(f"计算失败：{str(e)}")