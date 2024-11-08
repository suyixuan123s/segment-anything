# import yaml
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import open3d as o3d  # 用于处理点云
# from YOLOv9.capture_detect_results_confidence import class_ids, bboxes, conf_scores
# from segment_anything import sam_model_registry, SamPredictor
#
# # ----------------- 初始化 SAM -----------------
# sam_checkpoint = r"E:\ABB\segment-anything\weights\sam_vit_h_4b8939.pth"
# model_type = "vit_h"
# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# predictor = SamPredictor(sam)
#
# # ----------------- 加载图像和深度图 -----------------
# image_path = r'E:\ABB\AI\yolov9\data\data_realsense\color_image_20241014-112403.jpg'
# depth_path = r'E:\ABB\AI\yolov9\data\data_realsense\depth_image_20241014-112403.png'  # 深度图路径
# image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
# depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # 读取深度图
# predictor.set_image(image)
#
# # ----------------- 相机内参 -----------------
# # 假设你已经知道内参，可以使用你的相机内参
# fx, fy = 383.11187744140625, 383.11187744140625
# cx, cy = 325.05340576171875, 242.58470153808594
#
# # ----------------- 加载 COCO 类名 -----------------
# with open(r'E:\ABB\segment-anything\data\coco-dataset.yaml', 'r', encoding='utf-8') as file:
#     coco_data = yaml.safe_load(file)
#     class_names = coco_data['names']
#
# # ----------------- 初始化颜色映射 -----------------
# color_map = {}
# for class_id in class_ids:
#     color_map[class_id] = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#
# # ----------------- 定义显示函数 -----------------
# def show_mask(mask, ax, color):
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
#     ax.imshow(mask_image, alpha=0.5)
#
# def show_box(box, label, conf_score, color, ax):
#     x0, y0 = box[0], box[1]
#     w, h = box[2] - box[0], box[3] - box[1]
#     rect = plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor='none', lw=2)
#     ax.add_patch(rect)
#
#     label_offset = 10
#     label_text = f'{label} {conf_score:.2f}'
#     ax.text(x0, y0 - label_offset, label_text, color='black', fontsize=10, va='top', ha='left',
#             bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', boxstyle='square,pad=0.4'))
#
# # ----------------- 点云提取函数 -----------------
# def extract_point_cloud(mask, depth_image, fx, fy, cx, cy):
#     points = []
#     h, w = mask.shape
#     for v in range(h):
#         for u in range(w):
#             if mask[v, u] > 0:  # 如果像素在掩码内
#                 z = depth_image[v, u] / 1000.0  # 深度值，假设深度图单位为毫米
#                 if z == 0:  # 忽略无效深度值
#                     continue
#                 x = (u - cx) * z / fx
#                 y = (v - cy) * z / fy
#                 points.append([x, y, z])
#     return np.array(points)
#
# # ----------------- 显示图像和分割结果 -----------------
# plt.figure(figsize=(10, 10))
# ax = plt.gca()
# plt.imshow(image)
#
# # 点云列表，用于后面保存
# all_points = []
#
# # 遍历每个边界框并生成相应的掩码
# for class_id, bbox, conf in zip(class_ids, bboxes, conf_scores):
#     class_name = class_names[class_id]
#     color = color_map[class_id]
#     input_box = np.array(bbox)
#
#     # 使用 SAM 生成当前边界框的掩码
#     masks, _, _ = predictor.predict(
#         point_coords=None,
#         point_labels=None,
#         box=input_box,
#         multimask_output=False,
#     )
#
#     # 显示掩码和边界框
#     show_mask(masks[0], ax, color=color)
#     show_box(bbox, class_name, conf, color, ax)
#
#     # 提取分割区域的点云
#     points = extract_point_cloud(masks[0], depth_image, fx, fy, cx, cy)
#     all_points.append(points)
#
# # 显示最终的图像
# plt.axis('off')
# plt.show()
#
# # 将点云保存为 .ply 文件
# if len(all_points) > 0:
#     all_points = np.vstack(all_points)  # 合并所有点云
#     # 创建 Open3D 点云对象
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(all_points)
#
#     # 修改保存路径为 'E:\ABB\segment-anything\point_cloud_data'
#     save_path = r'E:\ABB\segment-anything\point_cloud_data\point_cloud.ply'
#
#     # 尝试保存点云文件
#     o3d.io.write_point_cloud(save_path, pcd)
#     print(f"点云已保存到 {save_path}")
#
import yaml
import cv2
import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt
from YOLOv9.capture_detect_results_confidence import class_ids, bboxes, conf_scores
from segment_anything import sam_model_registry, SamPredictor

# ----------------- 初始化 SAM -----------------
sam_checkpoint = r"E:\ABB\segment-anything\weights\sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

# ----------------- 加载图像和深度图 -----------------
image_path = r'E:\ABB\AI\yolov9\data\data_realsense\color_image_20241016-215422.jpg'
depth_path = r'E:\ABB\AI\yolov9\data\data_realsense\depth_image_20241016-215422.png'
image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # 读取深度图
predictor.set_image(image)

# ----------------- 相机内参 -----------------
# 假设你已经知道内参，可以使用你的相机内参
fx, fy = 606.626525878906, 606.6566772460938  # 焦距
cx, cy = 324.2806701660156, 241.14862060546875  # 光心位置

# ----------------- 加载 COCO 类名 -----------------
with open(r'E:\ABB\segment-anything\data\coco-dataset.yaml', 'r', encoding='utf-8') as file:
    coco_data = yaml.safe_load(file)
    class_names = coco_data['names']

# ----------------- 初始化颜色映射 -----------------
color_map = {}
for class_id in class_ids:
    color_map[class_id] = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)

# ----------------- 定义显示函数 -----------------
def show_mask(mask, ax, color):
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
    ax.imshow(mask_image, alpha=0.5)

def show_box(box, label, conf_score, color, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    rect = plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor='none', lw=2)
    ax.add_patch(rect)

    label_offset = 10
    label_text = f'{label} {conf_score:.2f}'
    ax.text(x0, y0 - label_offset, label_text, color='black', fontsize=10, va='top', ha='left',
            bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', boxstyle='square,pad=0.4'))

# ----------------- 点云提取函数 -----------------
def extract_point_cloud(mask, depth_image, fx, fy, cx, cy, label_id):
    points = []
    labels = []
    h, w = mask.shape
    for v in range(h):
        for u in range(w):
            if mask[v, u] > 0:  # 如果像素在掩码内
                z = depth_image[v, u] / 1000.0  # 深度值，假设深度图单位为毫米
                if z == 0:  # 忽略无效深度值
                    continue
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])
                labels.append(label_id)  # 为每个点附加类标签
    return np.array(points), np.array(labels)

# ----------------- 显示图像和分割结果 -----------------
plt.figure(figsize=(10, 10))
ax = plt.gca()
plt.imshow(image)

# 点云列表，用于后面保存
all_points = []
all_labels = []

# 遍历每个边界框并生成相应的掩码
for class_id, bbox, conf in zip(class_ids, bboxes, conf_scores):
    class_name = class_names[class_id]
    color = color_map[class_id]
    input_box = np.array(bbox)

    # 使用 SAM 生成当前边界框的掩码
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box,
        multimask_output=False,
    )

    # 显示掩码和边界框
    show_mask(masks[0], ax, color=color)
    show_box(bbox, class_name, conf, color, ax)

    # 提取分割区域的点云
    points, labels = extract_point_cloud(masks[0], depth_image, fx, fy, cx, cy, class_id)
    all_points.append(points)
    all_labels.append(labels)

# 显示最终的图像
plt.axis('off')
plt.show()

# 将点云保存为 .ply 文件
if len(all_points) > 0:
    all_points = np.vstack(all_points)  # 合并所有点云
    all_labels = np.hstack(all_labels)  # 合并所有标签

    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)

    # 将标签信息作为点的颜色，标准化为 [0, 1] 的范围，用于可视化
    label_colors = all_labels / len(class_names)  # 将标签映射到 [0, 1] 范围
    colors = np.zeros((len(all_labels), 3))  # 初始化颜色矩阵
    colors[:, 0] = label_colors  # 这里只用了标签映射来控制红色通道，你可以根据需要调整

    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 修改保存路径为 'E:\ABB\segment-anything\point_cloud_data'
    save_dir = r'E:\ABB\segment-anything\point_cloud_data'
    os.makedirs(save_dir, exist_ok=True)  # 如果文件夹不存在则创建
    save_path = os.path.join(save_dir, 'point_cloud_with_labels.ply')

    # 尝试保存点云文件
    o3d.io.write_point_cloud(save_path, pcd)
    print(f"点云已保存到 {save_path}")
