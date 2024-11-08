# import yaml
# from YOLOv9.capture_detect_results_confidence import class_ids, bboxes, conf_scores
# from segment_anything import sam_model_registry, SamPredictor
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import os
#
# # ----------------- 初始化 SAM -----------------
# # 加载预训练模型
# sam_checkpoint = r"E:\ABB\segment-anything\weights\sam_vit_h_4b8939.pth"  # 替换为实际路径
# model_type = "vit_h"
# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# predictor = SamPredictor(sam)
#
# # ----------------- 加载图像并设置图像 -----------------
# image_path = r'E:\ABB\AI\yolov9\data\data_realsense\color_image_20241014-112403.jpg'  # 替换为实际路径
# image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
# predictor.set_image(image)
#
# # ----------------- 加载 COCO 类名 -----------------
# with open(r'E:\ABB\segment-anything\data\coco-dataset.yaml', 'r', encoding='utf-8') as file:
#     coco_data = yaml.safe_load(file)
#     class_names = coco_data['names']  # 使用 YAML 文件中的 'names' 部分
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
#     ax.imshow(mask_image, alpha=0.5)  # 设置透明度为0.5
#
# def show_box(box, label, conf_score, color, ax):
#     x0, y0 = box[0], box[1]
#     w, h = box[2] - box[0], box[3] - box[1]
#     rect = plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor='none', lw=2)
#     ax.add_patch(rect)
#
#     label_offset = 10
#
#     # 构造带有类名和置信度分数的标签
#     label_text = f'{label} {conf_score:.2f}'
#
#     ax.text(x0, y0 - label_offset, label_text, color='black', fontsize=10, va='top', ha='left',
#             bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', boxstyle='square,pad=0.4'))
#
# # ----------------- 生成保存目录 -----------------
# save_dir = r"E:\ABB\AI\yolov9\segment\output"
# os.makedirs(save_dir, exist_ok=True)
#
# # ----------------- 生成结果图像并保存 -----------------
# # 创建一个用于合并掩码的图像
# mask_combined = np.zeros_like(image, dtype=np.uint8)  # 初始化为全黑的图像
# info_file_path = os.path.join(save_dir, 'detection_info.txt')  # 保存类别和置信度信息的文本文件
#
# with open(info_file_path, 'w') as f_info:
#     # 遍历每个边界框并生成相应的掩码
#     for class_id, bbox, conf in zip(class_ids, bboxes, conf_scores):
#         class_name = class_names[class_id]  # 获取类名
#         color = color_map[class_id]
#         input_box = np.array(bbox)
#
#         # 使用 SAM 生成当前边界框的掩码
#         masks, _, _ = predictor.predict(
#             point_coords=None,
#             point_labels=None,
#             box=input_box,
#             multimask_output=False,
#         )
#
#         # 将当前掩码添加到合并掩码图像中
#         mask = masks[0].astype(np.uint8)  # 转换为uint8类型
#         color_mask = np.zeros_like(image)  # 创建一个空白图像
#         color_mask[mask == 1] = (color[:3] * 255).astype(np.uint8)  # 将掩码上色
#         mask_combined = cv2.addWeighted(mask_combined, 1.0, color_mask, 0.5, 0)  # 合并掩码
#
#         # 保存检测到的类别和置信度信息到文本文件
#         f_info.write(f"Class: {class_name}, Confidence: {conf:.2f}, BBox: {bbox}\n")
#
# # 保存生成的掩码图像
# mask_image_path = os.path.join(save_dir, 'segmentation_mask.png')
# cv2.imwrite(mask_image_path, cv2.cvtColor(mask_combined, cv2.COLOR_RGB2BGR))
# print(f"分割掩码图像已保存到: {mask_image_path}")
# print(f"检测信息已保存到: {info_file_path}")
#
# # 显示最终的图像
# plt.figure(figsize=(10, 10))
# plt.imshow(mask_combined)
# plt.axis('off')
# plt.show()



import yaml
from YOLOv9.capture_detect_results_confidence import class_ids, bboxes, conf_scores
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------- 初始化 SAM -----------------
# 加载预训练模型
sam_checkpoint = r"E:\ABB\segment-anything\weights\sam_vit_h_4b8939.pth"  # 替换为实际路径
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

# ----------------- 加载图像并设置图像 -----------------
image_path = r'E:\ABB\segment-anything\output_individual\YOLOv9\exp\color_image_20241022-143158.jpg'  # 替换为实际路径
image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
predictor.set_image(image)

# ----------------- 加载 COCO 类名 -----------------
with open(r'E:\ABB\segment-anything\data\coco-dataset.yaml', 'r', encoding='utf-8') as file:
    coco_data = yaml.safe_load(file)
    class_names = coco_data['names']  # 使用 YAML 文件中的 'names' 部分

# ----------------- 初始化颜色映射 -----------------
color_map = {}
for class_id in class_ids:
    color_map[class_id] = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)

# ----------------- 定义显示函数 -----------------
def show_mask(mask, ax, color):
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
    ax.imshow(mask_image, alpha=0.5)  # 设置透明度为0.5

def show_box(box, label, conf_score, color, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    rect = plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor='none', lw=2)
    ax.add_patch(rect)

    label_offset = 10

    # 构造带有类名和置信度分数的标签
    label_text = f'{label} {conf_score:.2f}'

    ax.text(x0, y0 - label_offset, label_text, color='black', fontsize=10, va='top', ha='left',
            bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', boxstyle='square,pad=0.4'))

# ----------------- 生成保存目录 -----------------
save_dir = r"E:\ABB\segment-anything\YOLOv9\output"
os.makedirs(save_dir, exist_ok=True)

# ----------------- 生成结果图像并保存 -----------------
# 创建一个用于合并掩码的图像
mask_combined = np.zeros_like(image, dtype=np.uint8)  # 初始化为全黑的图像
info_file_path = os.path.join(save_dir, 'detection_info.txt')  # 保存类别和置信度信息的文本文件

with open(info_file_path, 'w') as f_info:
    # 遍历每个边界框并生成相应的掩码
    for class_id, bbox, conf in zip(class_ids, bboxes, conf_scores):
        class_name = class_names[class_id]  # 获取类名
        color = color_map[class_id]
        input_box = np.array(bbox)

        # 使用 SAM 生成当前边界框的掩码
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=False,
        )

        # 将当前掩码添加到合并掩码图像中
        mask = masks[0].astype(np.uint8)  # 转换为uint8类型
        color_mask = np.zeros_like(image)  # 创建一个空白图像
        color_mask[mask == 1] = (color[:3] * 255).astype(np.uint8)  # 将掩码上色
        mask_combined = cv2.addWeighted(mask_combined, 1.0, color_mask, 0.5, 0)  # 合并掩码

        # 保存检测到的类别和置信度信息到文本文件
        f_info.write(f"Class: {class_name}, Confidence: {conf:.2f}, BBox: {bbox}\n")

        # 保存单个物体的掩码图像
        object_mask_image = cv2.bitwise_and(image, image, mask=mask)
        object_save_path = os.path.join(save_dir, f'{class_name}_object_{conf:.2f}.png')
        cv2.imwrite(object_save_path, cv2.cvtColor(object_mask_image, cv2.COLOR_RGB2BGR))
        print(f"单个物体掩码图像已保存到: {object_save_path}")

# 保存生成的掩码图像
mask_image_path = os.path.join(save_dir, 'segmentation_mask.png')
cv2.imwrite(mask_image_path, cv2.cvtColor(mask_combined, cv2.COLOR_RGB2BGR))
print(f"分割掩码图像已保存到: {mask_image_path}")
print(f"检测信息已保存到: {info_file_path}")

# 显示最终的图像
plt.figure(figsize=(10, 10))
plt.imshow(mask_combined)
plt.axis('off')
plt.show()
