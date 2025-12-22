import argparse
import math

import trimesh
from trimesh import Trimesh, transformations

import numpy as np
import gc
import subprocess
import os
import re
import tempfile


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

from sklearn.neighbors import NearestNeighbors

# Chamfer距离计算
def chamfer_distance(mesh1, mesh2, num_points=2048):
    # 均匀采样点云
    points1 = mesh1.sample(num_points)
    points2 = mesh2.sample(num_points)

    # 双向最近邻
    nbrs1 = NearestNeighbors(n_neighbors=1).fit(points1)
    dists2, _ = nbrs1.kneighbors(points2)
    nbrs2 = NearestNeighbors(n_neighbors=1).fit(points2)
    dists1, _ = nbrs2.kneighbors(points1)

    chamfer = np.mean(dists1) + np.mean(dists2)
    return chamfer

# 计算单个IOU
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

# 包围盒相似度
def bbox_overlap_similarity(mesh1, mesh2):
    bb1 = mesh1.bounding_box.bounds  # shape (2, 3)
    bb2 = mesh2.bounding_box.bounds

    # 计算交集体积
    inter_min = np.maximum(bb1[0], bb2[0])
    inter_max = np.minimum(bb1[1], bb2[1])
    inter_vol = np.prod(np.clip(inter_max - inter_min, 0, None))

    vol1 = np.prod(bb1[1] - bb1[0])
    vol2 = np.prod(bb2[1] - bb2[0])
    union_vol = vol1 + vol2 - inter_vol

    if union_vol == 0:
        return 1.0
    return inter_vol / union_vol

# 表面积相似度
def surface_area_similarity(mesh1, mesh2):
    a1 = mesh1.area
    a2 = mesh2.area
    if max(a1, a2) == 0:
        return 1.0
    return 1.0 - abs(a1 - a2) / max(a1, a2)

# 体积相似度
def volume_similarity(mesh1, mesh2):
    v1 = abs(mesh1.volume)
    v2 = abs(mesh2.volume)
    if max(v1, v2) == 0:
        return 1.0
    return 1.0 - abs(v1 - v2) / max(v1, v2)

def calculate_3d_iou(mesh1, mesh2):
    """
    计算两个STL模型的3D IOU（交并比）
    将第一个模型旋转到x,y,z轴三个方向，计算与第二个模型的IOU，取最大值
    """
    # # 1. 读取STL文件
    # mesh1 = trimesh.load(stl_path1)
    # mesh2 = trimesh.load(stl_path2)
    
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

def center_mesh(mesh: Trimesh) -> Trimesh:
    """Center both meshes to origin by moving their center to (0,0,0)."""
    mesh = mesh.copy()
    mesh.apply_transform(transformations.translation_matrix(-mesh.center_mass))
    return mesh

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

def extract_cot(solution_str):
    """Extract the equation from the solution string."""

    answer_pattern = r"<think>(.*?)</think>"
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

# def convert_openscad_to_stl(scad_file, stl_file, openscad_path="/usr/bin/openscad-nightly"):
#     """将OpenSCAD文件转换为STL格式"""
#     try:
#         subprocess.run(
#             [openscad_path, "-o", stl_file,"--backend", "Manifold", scad_file],
#             check=True,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             text=True,
#             timeout=10
#         )
#         # print(f"成功将 {scad_file} 转换为 {stl_file}")
#         return True
#     except subprocess.CalledProcessError as e:
#         # print(f"转换失败: {e.stderr}")
#         return False
#     except FileNotFoundError:
#         # print(f"未找到OpenSCAD可执行文件，请检查路径是否正确: {openscad_path}")
#         return False
#     except subprocess.TimeoutExpired:
#         # print(f"转换超时: {scad_file}")
#         return False

import resource 
def set_resource_limits():
    # 设置CPU时间限制（秒），软限制和硬限制
    # 超过软限制会收到信号，超过硬限制会被终止
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))  # 软限制10秒，硬限制15秒
    
    # 设置内存限制（字节）
    # 这里设置为1GB内存限制
    mem_limit = 1 * 1024 * 1024 * 1024  # 1GB
    resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))  # 虚拟内存限制


def convert_openscad_to_stl(scad_file, stl_file, openscad_path="/home/yc27979/xa/openscad_verl/openscad"):
    """将OpenSCAD文件转换为STL格式"""
    try:
        subprocess.run(
            ["/home/yc27979/xa/openscad_verl/examples/my_trainer/run_openscad.sh", openscad_path, stl_file, scad_file],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            preexec_fn=set_resource_limits  # 子进程启动前设置资源限制
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


def apply_length_based_reward_adjustment(iou_reward, token_length, max_length):
    """
    Apply length-based reward adjustments to IOU scores.
    
    Args:
        iou_reward (float): The original IOU reward score
        solution_length (int): Length of the current answer
        max_length (int): Length of the longest answer in the batch
        
    Returns:
        float: Adjusted reward score based on length
    """
    if max_length == 0:
        return iou_reward
    # print("token_length:"+ str(token_length))
    # print("max_length:" + str(max_length))
    
    length_ratio = token_length / max_length
    # print("length_ratio: " + str(length_ratio))


    # print("###################################################")
    
    if token_length == max_length:
        return iou_reward - 0.5
    # elif 0.7 <= length_ratio <= 0.99:
    #     if iou_reward < 0.0001:
    #         return iou_reward - 0.1
    #     else:
    #         return iou_reward
    else:
        return iou_reward

