"""
Author: Yixuan Su
Date: 2024/11/21 12:08
File: YOLOv9_SAM_point_cloud_extract_transformation.py
Description:
"""

import os
import torch
import cv2
import numpy as np
import pyrealsense2 as rs
from pathlib import Path
from models.common import DetectMultiBackend
from segment_anything import sam_model_registry, SamPredictor
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
import pandas as pd

import open3d as o3d

# 路径
yolo_weights = r'E:\ABB\segment-anything\runs\train\exp19\weights\best.pt'  # YOLO 权重文件路径
source = r'color_image_20241026-194402.jpg'  # 输入图像路径
depth_image_path = r'depth_image_20241026-194402.png'  # 深度图像路径
sam_checkpoint = r"E:\ABB\segment-anything\weights\sam_vit_h_4b8939.pth"  # SAM 模型权重路径
model_type = "vit_h"  # SAM 模型类型

# 用于从相机到仿真环境的坐标转换参数
alpha, beta, gamma = -148.0, -0.4, -178.0  # 旋转角度
tx, ty, tz = 0.525, 0.76, 1.25  # 平移量

# 初始化 RealSense 相机管道并获取相机内参
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)  # 配置深度流
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # 配置彩色流
profile = pipeline.start(config)

# 对齐深度图和彩色图
align = rs.align(rs.stream.color)

# 获取深度相机的内参
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
fx, fy, cx, cy = depth_intrinsics.fx, depth_intrinsics.fy, depth_intrinsics.ppx, depth_intrinsics.ppy
intrinsics = [fx, fy, cx, cy]  # 保存相机内参


# YOLO 检测函数
def yolo_detect(img_path):
    device = select_device('')  # 选择设备（例如 GPU）
    model = DetectMultiBackend(yolo_weights, device=device)  # 加载 YOLO 模型
    img0 = cv2.imread(img_path)  # 读取输入图像
    img = cv2.resize(img0, (640, 480))  # 调整图像大小
    img = torch.from_numpy(img).to(device).float() / 255.0  # 转换为张量
    img = img.permute(2, 0, 1).unsqueeze(0)
    pred = model(img)  # 进行检测
    pred = non_max_suppression(pred, 0.25, 0.45)  # 应用非极大值抑制
    results = []
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                results.append({
                    "bbox": [int(x) for x in xyxy],  # 检测框坐标
                    "confidence": float(conf),  # 置信度
                    "class": int(cls),  # 类别索引
                    "class_name": model.names[int(cls)]  # 类别名称
                })
    return img0, results  # 返回原始图像和检测结果


# SAM 分割函数
def sam_segment(img0, bbox):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)  # 加载 SAM 模型
    predictor = SamPredictor(sam)
    predictor.set_image(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))  # 设置图像
    masks, _, _ = predictor.predict(point_coords=None, point_labels=None, box=np.array(bbox), multimask_output=False)
    return masks[0]  # 返回分割掩码


# 生成用于相机到仿真环境的转换矩阵
def get_transformation_matrix(alpha_deg, beta_deg, gamma_deg, tx, ty, tz):
    alpha, beta, gamma = np.radians([alpha_deg, beta_deg, gamma_deg])  # 将角度转换为弧度
    Rx = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
    rotation_matrix = Rz @ Ry @ Rx  # 计算旋转矩阵
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix  # 将旋转矩阵设置为转换矩阵的一部分
    transformation_matrix[:3, 3] = [tx, ty, tz]  # 设置平移部分
    return transformation_matrix  # 返回转换矩阵


# 提取点云并保存为 .ply 文件，同时返回中心点坐标
def extract_point_cloud_from_mask_ply(depth_image, mask, intrinsics, transformation_matrix, save_path):
    points = []
    colors = []
    h, w = depth_image.shape
    fx, fy, cx, cy = intrinsics
    for y in range(h):
        for x in range(w):
            if mask[y, x]:
                depth = depth_image[y, x] * 0.001  # 将深度值转换为米
                if depth > 0:
                    X = (x - cx) * depth / fx
                    Y = (y - cy) * depth / fy
                    Z = depth
                    camera_coords = np.array([X, Y, Z, 1])  # 生成齐次坐标
                    world_coords = transformation_matrix @ camera_coords  # 转换为仿真坐标
                    points.append(world_coords[:3])

                    # 获取对应像素的颜色
                    color = color_image[y, x] / 255.0  # 将 BGR 颜色值归一化到 [0, 1]
                    colors.append(color)

    # 使用 Open3D 保存为带颜色的 PLY 文件
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)  # 将颜色信息添加到点云
    o3d.io.write_point_cloud(save_path, point_cloud)  # 保存点云文件

    # 计算点云的中心点坐标
    sim_coords = np.mean(points, axis=0) if points else [0, 0, 0]
    return sim_coords.round(2)  # 返回中心点坐标


if __name__ == "__main__":
    # 设置转换矩阵
    transformation_matrix = get_transformation_matrix(alpha, beta, gamma, tx, ty, tz)
    img0, detections = yolo_detect(source)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    color_image = cv2.imread(source)  # 加载彩色图像用于颜色信息

    # 主结果文件夹路径
    save_folder = r'SAM'
    os.makedirs(save_folder, exist_ok=True)

    complete_detection_img = img0.copy()  # 完整的检测图像
    complete_segmentation_img = img0.copy()  # 完整的分割图像

    # 处理每个检测到的物体
    for idx, detection in enumerate(detections):
        bbox = detection["bbox"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]

        # 为每个类别创建单独的文件夹
        object_folder = os.path.join(save_folder, class_name, f"{class_name}_{idx}")
        os.makedirs(object_folder, exist_ok=True)

        # 使用 SAM 进行分割
        mask = sam_segment(img0, bbox)

        # 提取点云并保存为 .ply 文件，获取中心点坐标
        ply_save_path = os.path.join(object_folder, "point_cloud.ply")
        sim_coords = extract_point_cloud_from_mask_ply(depth_image, mask, intrinsics, transformation_matrix,
                                                       ply_save_path)
        sim_coords_text = f"Sim XYZ: {sim_coords}"

        # 创建带有边界框和仿真信息的单独注释图像
        x0, y0, x1, y1 = bbox
        annotated_img = img0.copy()
        cv2.rectangle(annotated_img, (x0, y0), (x1, y1), (0, 255, 0), 2)  # 绘制边界框
        cv2.circle(annotated_img, (int((x0 + x1) / 2), int((y0 + y1) / 2)), 5, (255, 0, 0), -1)  # 绘制中心点
        cv2.putText(annotated_img, f"{class_name} {confidence:.2f}", (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2)
        cv2.putText(annotated_img, sim_coords_text, (x0, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imwrite(os.path.join(object_folder, "annotated.jpg"), annotated_img)  # 保存标注图像

        # 在完整的检测和分割图像上添加信息
        cv2.rectangle(complete_detection_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.circle(complete_detection_img, (int((x0 + x1) / 2), int((y0 + y1) / 2)), 5, (255, 0, 0), -1)
        cv2.putText(complete_detection_img, f"{class_name} {confidence:.2f}", (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)
        cv2.putText(complete_detection_img, sim_coords_text, (x0, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                    2)

        # 在完整分割图像上应用掩码
        mask_area = mask > 0
        colored_mask = np.zeros_like(complete_segmentation_img)
        colored_mask[mask_area] = [255, 0, 0]
        complete_segmentation_img[mask_area] = cv2.addWeighted(complete_segmentation_img[mask_area], 0.7,
                                                               colored_mask[mask_area], 0.3, 0)
        cv2.rectangle(complete_segmentation_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.circle(complete_segmentation_img, (int((x0 + x1) / 2), int((y0 + y1) / 2)), 5, (255, 0, 0), -1)
        cv2.putText(complete_segmentation_img, sim_coords_text, (x0, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 0), 2)

        # 保存单个分割掩码
        individual_img = img0.copy()
        individual_img[~mask_area] = 0
        cv2.imwrite(os.path.join(object_folder, "segmented.jpg"), individual_img)  # 保存分割图像

        # 将类别信息和仿真坐标打印到控制台
        print(f"Class: {class_name}, Confidence: {confidence:.2f}, Simulation Position: {sim_coords_text}")

    # 将完整的检测和分割图像保存到主文件夹中
    cv2.imwrite(os.path.join(save_folder, "complete_detection.jpg"), complete_detection_img)
    cv2.imwrite(os.path.join(save_folder, "complete_segmentation.jpg"), complete_segmentation_img)
