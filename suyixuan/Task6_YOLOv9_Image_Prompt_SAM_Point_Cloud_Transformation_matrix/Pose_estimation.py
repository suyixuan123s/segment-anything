"""
Author: Yixuan Su
Date: 2024/12/02 10:02
File: Pose_estimation.py
Description: 
"""

import open3d as o3d
import numpy as np

# 外参矩阵（从相机坐标系到世界坐标系的变换矩阵）
extrinsic_matrix = np.array([
    [-0.999366, -0.033294, 0.012577, 0.525000],
    [-0.034899, 0.847402, -0.529803, 0.760000],
    [0.006981, -0.529906, -0.848027, 1.250000],
    [0.000000, 0.000000, 0.000000, 1.000000]
])


# 读取点云文件
def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd


# 计算点云的中心点
def calculate_center_of_point_cloud(pcd):
    points = np.asarray(pcd.points)
    center = points.mean(axis=0)  # 计算点云的几何中心
    return center


# 创建一个红色小球表示中心点
def add_center_point_to_point_cloud(center):
    # 创建一个小球，表示点云的中心
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)  # 小球半径
    sphere.translate(center)  # 将小球放置到中心点
    sphere.paint_uniform_color([1, 0, 0])  # 红色
    return sphere


# 将相机坐标系中的中心点转换到世界坐标系
def transform_to_world_coordinate(center, extrinsic_matrix):
    # 使用外参矩阵将中心点从相机坐标系转换到世界坐标系
    center_homogeneous = np.append(center, 1)  # 转换为齐次坐标
    transformed_center = np.dot(extrinsic_matrix, center_homogeneous)  # 变换
    return transformed_center[:3]  # 返回转换后的 3D 坐标


# 输出位姿信息（位置 + 姿态）
def print_pose(center):
    print(f"红球位置（世界坐标系）：{center}")

    # 旋转部分：假设红球没有旋转，姿态为单位旋转矩阵
    rotation_matrix = np.eye(3)  # 单位旋转矩阵，表示没有旋转
    print(f"红球姿态（世界坐标系，旋转矩阵）：\n{rotation_matrix}")


# 显示点云和中心点
def visualize_point_cloud_with_center(pcd, center_sphere):
    o3d.visualization.draw_geometries([pcd, center_sphere], window_name="Point Cloud with Center Highlight",
                                      width=800, height=600)


# 点云文件路径
point_cloud_path = r"E:\ABB\segment-anything\suyixuan\Task6_YOLOv9_Image_Prompt_SAM_Point_Cloud_Transformation_matrix\SAM\5ML_sorting_tube_rack\5ML_sorting_tube_rack_6\point_cloud.ply"

# 加载点云
pcd = load_point_cloud(point_cloud_path)

# 计算点云的几何中心
center = calculate_center_of_point_cloud(pcd)
print(f"点云中心点坐标（相机坐标系）：{center}")

# 使用外参矩阵将点云的中心点从相机坐标系转换到世界坐标系
world_center = transform_to_world_coordinate(center, extrinsic_matrix)
print(f"点云中心点坐标（世界坐标系）：{world_center}")

# 输出红球的位姿信息
print_pose(world_center)

# 创建一个红色小球，表示点云的中心
center_sphere = add_center_point_to_point_cloud(center)

# 显示点云和中心点
visualize_point_cloud_with_center(pcd, center_sphere)
