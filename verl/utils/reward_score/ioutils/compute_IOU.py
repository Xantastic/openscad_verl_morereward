import trimesh
import numpy as np
import gc


def rotate_mesh_to_axis(mesh, axis):
    """
    将模型旋转到指定轴方向
    axis: 'x', 'y', 'z' 表示要旋转到的轴
    """
    # 获取模型的主方向（通过PCA）
    vertices = mesh.vertices
    center = vertices.mean(axis=0)
    centered_vertices = vertices - center
    
    # 计算协方差矩阵
    cov_matrix = np.cov(centered_vertices.T)
    
    # 计算特征向量
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # 找到最大特征值对应的特征向量（主方向）
    main_axis = eigenvectors[:, np.argmax(eigenvalues)]
    
    # 根据目标轴确定旋转
    if axis == 'x':
        target_axis = np.array([1, 0, 0])
    elif axis == 'y':
        target_axis = np.array([0, 1, 0])
    elif axis == 'z':
        target_axis = np.array([0, 0, 1])
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    
    # 计算旋转轴和旋转角度
    rotation_axis = np.cross(main_axis, target_axis)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    
    if rotation_axis_norm < 1e-6:
        # 已经平行，不需要旋转
        return mesh
    
    rotation_axis = rotation_axis / rotation_axis_norm
    
    # 计算旋转角度
    cos_angle = np.dot(main_axis, target_axis)
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    
    # 创建旋转矩阵
    rotation_matrix = trimesh.transformations.rotation_matrix(angle, rotation_axis)
    
    # 应用旋转
    rotated_mesh = mesh.copy()
    rotated_mesh.apply_transform(rotation_matrix)
    
    return rotated_mesh


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
    将第一个模型旋转到x,y,z轴三个方向，计算与第二个模型的IOU，取最大值
    """
    # 1. 读取STL文件
    mesh1 = trimesh.load(stl_path1)
    mesh2 = trimesh.load(stl_path2)
    
    try:
        # 2. 将mesh1旋转到三个主轴方向
        mesh1_x = rotate_mesh_to_axis(mesh1, 'x')
        mesh1_y = rotate_mesh_to_axis(mesh1, 'y')
        mesh1_z = rotate_mesh_to_axis(mesh1, 'z')

        # 3. 计算三个方向的IOU
        try:
            iou_x = calculate_single_iou(mesh1_x, mesh2)
            iou_y = calculate_single_iou(mesh1_y, mesh2)
            iou_z = calculate_single_iou(mesh1_z, mesh2)
            
            # 取最大值
            max_iou = max(iou_x, iou_y, iou_z)
            return max_iou
            
        except Exception as e:
            # 如果旋转后计算失败，尝试原始方向
            try:
                return calculate_single_iou(mesh1, mesh2)
            except Exception:
                return 0.0
                
    finally:
        # Clean up mesh objects to prevent memory leaks
        for mesh in [mesh1, mesh2]:
            if hasattr(mesh, 'close'):
                mesh.close()
            del mesh
        
        # Clean up rotated meshes if they exist
        for mesh_var in ['mesh1_x', 'mesh1_y', 'mesh1_z']:
            if mesh_var in locals():
                mesh_obj = locals()[mesh_var]
                if hasattr(mesh_obj, 'close'):
                    mesh_obj.close()
                del mesh_obj
        
        # Force garbage collection
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