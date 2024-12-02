"""
Author: Yixuan Su
Date: 2024/11/21 12:08
File: Analyze_point_cloud.py
Description:
"""

import open3d as o3d
import numpy as np

# 点云文件路径
point_cloud_path = r'E:\ABB\segment-anything\suyixuan\Task6_YOLOv9_Image_Prompt_SAM_Point_Cloud_Transformation_matrix\SAM\5ML_sorting_tube_rack\5ML_sorting_tube_rack_6\point_cloud.ply'

# 加载点云文件（保留颜色信息）
point_cloud = o3d.io.read_point_cloud(point_cloud_path)

# 检查点云是否包含颜色信息
if point_cloud.has_colors():
    print("点云包含颜色信息，将显示带颜色的点云。")
else:
    print("点云不包含颜色信息，将显示为无色点云。")

# # 显示点云（保留原始颜色）
# o3d.visualization.draw_geometries([point_cloud], window_name="Colored Point Cloud", width=800, height=600)

# 获取点云的所有点坐标和颜色
points = np.asarray(point_cloud.points)
colors = np.asarray(point_cloud.colors) if point_cloud.has_colors() else None

# 计算点的数量
num_points = points.shape[0]

# 计算中心点（所有点的平均坐标）
center_point = points.mean(axis=0)

# # 将中心点添加到点云中并设为红色
# highlighted_points = np.vstack([points, center_point])
# highlighted_colors = np.vstack([colors, [1, 0, 0]])  # 红色高亮
#
# # 创建新的点云对象，包括原始点和中心点
# highlighted_point_cloud = o3d.geometry.PointCloud()
# highlighted_point_cloud.points = o3d.utility.Vector3dVector(highlighted_points)
# highlighted_point_cloud.colors = o3d.utility.Vector3dVector(highlighted_colors)

# 创建一个球体作为中心点高亮显示
center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)  # 设置球体的半径
center_sphere.translate(center_point)  # 将球体移动到中心点位置
center_sphere.paint_uniform_color([1, 0, 0])  # 将球体颜色设为红色

# 将原始点和球体一起显示
o3d.visualization.draw_geometries([point_cloud, center_sphere], window_name="Point Cloud with Center Highlight",
                                  width=800, height=600)

# 输出结果
print(f"点的数量: {num_points}")
print(f"中心点位置: {center_point}")

# # 显示带有高亮中心点的点云
# o3d.visualization.draw_geometries([highlighted_point_cloud], window_name="Point Cloud with Center Highlight", width=800, height=600)


# 返回点的数量、中心点和颜色信息（如果存在）
num_points, center_point, colors
