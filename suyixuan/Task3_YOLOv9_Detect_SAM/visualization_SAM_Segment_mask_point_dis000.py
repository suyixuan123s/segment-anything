"""
Author: Yixuan Su
Date: 2024/11/21 12:08
File: visualization_SAM_Segment_mask_point_dis000.py
Description:
"""

import cv2
import numpy as np
import open3d as o3d
import os
import csv

# ----------------- 文件路径 -----------------
image_path = r'E:\ABB\AI\yolov9\data\data_realsense\color_image_20241016-215422.jpg'
depth_path = r'E:\ABB\AI\yolov9\data\data_realsense\depth_image_20241016-215422.png'
mask_path = r'E:\ABB\AI\yolov9\segment\output\blood_tube_object_0.93.png'  # 掩码图像

# ----------------- 相机内参 -----------------
fx, fy = 606.626525878906, 606.6566772460938  # 焦距
cx, cy = 324.2806701660156, 241.14862060546875  # 光心位置

# ----------------- 读取图像 -----------------
image = cv2.imread(image_path)
depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # 读取深度图
mask_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # 读取掩码图像

# 确保掩码图像是二值化图像 (只有黑白，非 RGB)
mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
_, binary_mask = cv2.threshold(mask_image_gray, 1, 255, cv2.THRESH_BINARY)  # 阈值化处理

# ----------------- 保存深度信息到 CSV 文件 -----------------
csv_save_path = r'/depth_data/blood_tube_depth_info.csv'

# 创建保存深度数据的目录，如果不存在则创建
os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)

# 打开 CSV 文件进行写入
with open(csv_save_path, mode='w', newline='') as depth_file:
    csv_writer = csv.writer(depth_file)

    # 写入文件头信息
    csv_writer.writerow(["Pixel u", "Pixel v", "Depth (m)"])


    # ----------------- 提取掩码区域的三维点云 -----------------
    def extract_point_cloud(mask, depth_image, fx, fy, cx, cy, writer):
        points = []
        h, w = mask.shape
        for v in range(h):
            for u in range(w):
                z = depth_image[v, u] / 1000.0  # 深度信息从毫米转换为米
                if mask[v, u] > 0:  # 只处理掩码区域内的像素点
                    # 利用相机内参计算三维坐标 (X, Y, Z)
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    points.append([x, y, z])
                # 写入所有像素点的深度信息到 CSV 文件，包括深度为0的点
                writer.writerow([u, v, f"{z:.3f}"])
        return np.array(points)


    # 提取掩码区域的三维点云
    points = extract_point_cloud(binary_mask, depth_image, fx, fy, cx, cy, csv_writer)

# ----------------- 输出点云信息 -----------------
if len(points) > 0:
    print("提取的三维点云信息：")
    for point in points:
        print(f"X: {point[0]:.3f}, Y: {point[1]:.3f}, Z: {point[2]:.3f}")

    # ----------------- 保存点云到 PLY 文件 -----------------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    save_path = r'/point_cloud_data/blood_tube_point_cloud.ply'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    o3d.io.write_point_cloud(save_path, pcd)
    print(f"点云已保存到 {save_path}")
else:
    print("未提取到有效的点云")

# 可视化提取到的掩码图像
cv2.imshow('Mask', binary_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"所有像素点的深度信息已保存到 {csv_save_path}")
