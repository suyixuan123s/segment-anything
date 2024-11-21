"""
Author: Yixuan Su
Date: 2024/11/21 12:08
File: Capture_Point_cloud_Single_object.py
Description:
"""

import cv2
import numpy as np
import open3d as o3d

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


# ----------------- 提取掩码区域的三维点云 -----------------
def extract_point_cloud(mask, depth_image, fx, fy, cx, cy):
    points = []
    h, w = mask.shape
    for v in range(h):
        for u in range(w):
            if mask[v, u] > 0:  # 只处理掩码区域内的像素点
                z = depth_image[v, u] / 1000.0  # 深度信息从毫米转换为米
                if z == 0:  # 忽略无效深度
                    continue
                # 利用相机内参计算三维坐标 (X, Y, Z)
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])
    return np.array(points)


# 提取掩码区域的三维点云
points = extract_point_cloud(binary_mask, depth_image, fx, fy, cx, cy)

# ----------------- 输出点云信息 -----------------
if len(points) > 0:
    print("提取的三维点云信息：")
    for point in points:
        print(f"X: {point[0]:.3f}, Y: {point[1]:.3f}, Z: {point[2]:.3f}")

    # ----------------- 保存点云到 PLY 文件 -----------------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    save_path = r'/point_cloud_data/blood_tube_point_cloud.ply'
    o3d.io.write_point_cloud(save_path, pcd)
    print(f"点云已保存到 {save_path}")
else:
    print("未提取到有效的点云")

# 可视化提取到的掩码图像
cv2.imshow('Mask', binary_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
