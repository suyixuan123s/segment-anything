import os

import cv2
import numpy as np
import open3d as o3d
import yaml
from segment_anything import sam_model_registry, SamPredictor

# ----------------- 初始化 SAM -----------------
sam_checkpoint = r"E:\ABB\segment-anything\weights\sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

# ----------------- 加载图像和深度图 -----------------
image_path = r'E:\ABB\AI\yolov9\data\data_realsense\color_image_20241014-112403.jpg'
depth_path = r'E:\ABB\AI\yolov9\data\data_realsense\depth_image_20241014-112403.png'
image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # 读取深度图
predictor.set_image(image)

# ----------------- 相机内参 -----------------
fx, fy = 383.11187744140625, 383.11187744140625  # 焦距
cx, cy = 325.05340576171875, 242.58470153808594  # 光心位置

# ----------------- 加载 COCO 类名 -----------------
with open(r'E:\ABB\segment-anything\data\coco-dataset.yaml', 'r', encoding='utf-8') as file:
    coco_data = yaml.safe_load(file)
    class_names = coco_data['names']

# ----------------- 定义点云提取函数 -----------------
def extract_point_cloud(mask, depth_image, fx, fy, cx, cy):
    points = []
    h, w = mask.shape
    for v in range(h):
        for u in range(w):
            if mask[v, u] > 0:  # 只提取掩码区域内的点
                z = depth_image[v, u] / 1000.0  # 将深度值转换为米（假设深度图单位为毫米）
                if z == 0:  # 忽略无效深度
                    continue
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])
    return np.array(points)

# ----------------- 提取分割掩码并生成点云 -----------------
masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=np.array([0, 0, image.shape[1], image.shape[0]]),  # 使用整幅图像作为掩码边界
    multimask_output=False,
)

mask = masks[0].astype(np.uint8)  # 转换掩码为 uint8 类型
points = extract_point_cloud(mask, depth_image, fx, fy, cx, cy)  # 提取点云

# ----------------- 保存点云到文件 -----------------
if len(points) > 0:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    save_dir = r'E:\ABB\segment-anything\point_cloud_data'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'segmentation_point_cloud.ply')
    o3d.io.write_point_cloud(save_path, pcd)
    print(f"点云已保存到 {save_path}")
else:
    print("未提取到有效的点云")

# 可视化提取到的掩码和点云
cv2.imshow('Mask', mask * 255)
cv2.waitKey(0)
cv2.destroyAllWindows()
