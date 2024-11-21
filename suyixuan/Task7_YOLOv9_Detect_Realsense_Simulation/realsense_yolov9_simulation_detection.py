"""
Author: Yixuan Su
Date: 2024/11/21 12:08
File: realsense_yolov9_simulation_detection.py
Description:
"""

import os
import cv2
import numpy as np
import pandas as pd
import pyrealsense2 as rs
from suyixuan.Task1_YOLOv9_Detect.YOLOv9_Detect_API import DetectAPI

# 初始化 YOLOv9 模型
model = DetectAPI(weights=r'E:\ABB\segment-anything\suyixuan\exp19\weights\best.pt')

# 深度相机配置
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)

# 获取深度相机的内参
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

# 创建转换矩阵
alpha, beta, gamma = -148.0, -0.4, -178.0
tx, ty, tz = 0.525, 0.76, 1.25

# 类别名称映射（替换为你的 YOLO 模型的实际类别名称）
class_names = {
    0: 'blood_tube',
    1: '5ML_centrifuge_tube',
    2: '10ML_centrifuge_tube',
    3: '5ML_sorting_tube_rack',
    4: '10ML_sorting_tube_rack',
    5: 'centrifuge_open',
    6: 'centrifuge_close',
    7: 'refrigerator_open',
    8: 'refrigerator_close',
    9: 'operating_desktop',
    10: 'tobe_sorted_tube_rack',
    11: 'dispensing_tube_rack',
    12: 'sorting_tube_rack_base',
    13: 'tube_rack_storage_cabinet'
}


def get_transformation_matrix(alpha, beta, gamma, tx, ty, tz):
    alpha, beta, gamma = np.radians([alpha, beta, gamma])
    Rx = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
    rotation_matrix = Rz @ Ry @ Rx
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = [tx, ty, tz]
    return transformation_matrix


transformation_matrix = get_transformation_matrix(alpha, beta, gamma, tx, ty, tz)

# 加载 RGB 和深度图像
rgb_image_path = "../color_image_20241026-194402.jpg"
depth_image_path = "../depth_image_20241026-194402.png"
rgb_image = cv2.imread(rgb_image_path)
depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

# 深度图像转换比例
depth_scale = 0.001  # 假设深度图像的单位是毫米

# 检测物体并转换坐标
detections = model.detect([rgb_image])[1]

results_data = []
for det in detections:
    for *xyxy, conf, cls in det:
        cls = int(cls)
        class_name = class_names.get(cls, "Unknown")

        # 计算边界框的中心点
        x1, y1, x2, y2 = map(int, xyxy)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        depth = depth_image[center_y, center_x] * depth_scale

        # 转换为相机坐标
        cam_coords_center = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [center_x, center_y], depth)
        cam_coords_center = np.append(cam_coords_center, 1)

        # 转换为仿真坐标
        sim_coords_center = transformation_matrix @ cam_coords_center
        sim_coords_center = sim_coords_center[:3] * 1000  # 转换为毫米单位

        # 绘制边界框、类别、置信度
        cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(rgb_image, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 绘制中心点和标注仿真坐标信息
        cv2.circle(rgb_image, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.putText(rgb_image, f"Sim: {sim_coords_center}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                    2)

        # 记录检测结果
        results_data.append({
            'Class': class_name,
            'Confidence': conf,
            'Bounding Box': [x1, y1, x2, y2],
            'Camera XYZ': cam_coords_center[:3].tolist(),
            'Simulation XYZ': sim_coords_center.tolist()
        })

# 保存带有检测信息的图像
output_image_path = r"detected_simulation_image.jpg"
cv2.imwrite(output_image_path, rgb_image)

# 保存检测数据到 CSV 文件
csv_path = r'detection_results_simulation_coords.csv'
# 检查并创建路径中的目录
os.makedirs(os.path.dirname(csv_path), exist_ok=True)

df = pd.DataFrame(results_data)
df.to_csv(csv_path, index=False)
print(f"Results saved to {csv_path}")
print(f"Image with detections saved to {output_image_path}")
