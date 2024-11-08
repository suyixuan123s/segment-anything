'''

import os
import torch
import cv2
import numpy as np
from pathlib import Path
from models.common import DetectMultiBackend
from segment_anything import sam_model_registry, SamPredictor
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
import matplotlib.pyplot as plt

# YOLOv9 目标检测模型路径
yolo_weights = r'E:\ABB\segment-anything\runs\train\exp19\weights\best.pt'
source = r'E:\ABB\AI\Depth-Anything-V2\Inter_Realsence_D345_Datasets\color_image_20241022-143158.jpg'

# SAM 模型加载路径
sam_checkpoint = r"E:\ABB\segment-anything\weights\sam_vit_h_4b8939.pth"
model_type = "vit_h"

# ----------------- YOLOv9 目标检测 -----------------

def yolo_detect(img_path):
    device = select_device('')
    model = DetectMultiBackend(yolo_weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt

    # 读取图片
    img0 = cv2.imread(img_path)
    img = cv2.resize(img0, (640, 480))  # 将图像调整为 YOLOv9 模型输入尺寸
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # 归一化
    img = img.permute(2, 0, 1).unsqueeze(0)  # 调整维度以适应模型输入

    # 目标检测推理
    pred = model(img)
    pred = non_max_suppression(pred, 0.25, 0.45)

    # 获取检测结果
    results = []
    for i, det in enumerate(pred):
        if len(det):
            # 将边界框调整回原图尺寸
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                results.append({
                    "bbox": [int(x) for x in xyxy],  # 边界框坐标
                    "confidence": float(conf),  # 置信度
                    "class": int(cls),  # 类别ID
                    "class_name": names[int(cls)]  # 类别名称
                })
    return img0, results


# ----------------- SAM 目标分割 -----------------

def sam_segment(img, bbox):
    # 初始化 SAM 模型
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)

    # 转换图像颜色空间
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    # 将 YOLO 边界框传递给 SAM 进行分割
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.array(bbox),
        multimask_output=False
    )

    return masks[0]  # 返回分割掩码


# ----------------- 单独保存每个分割目标 -----------------

def save_individual_segment(img, mask, bbox, label, save_path):
    x0, y0, x1, y1 = bbox

    # 应用掩码到图像
    segmented_img = img.copy()
    mask_area = mask > 0
    segmented_img[~mask_area] = 0  # 将非目标区域置为黑色

    # 裁剪目标区域
    cropped_img = segmented_img[y0:y1, x0:x1]

    # 保存裁剪的图像
    cv2.imwrite(save_path, cropped_img)
    print(f"目标 {label} 的分割图像已保存: {save_path}")


# ----------------- 主函数 -----------------

if __name__ == "__main__":
    # YOLOv9 检测
    img, detections = yolo_detect(source)

    # 结果保存路径
    save_folder = r'E:\ABB\segment-anything\output_individual\SAM'
    os.makedirs(save_folder, exist_ok=True)

    # 遍历每个目标，使用 SAM 分割并保存结果
    for idx, detection in enumerate(detections):
        bbox = detection["bbox"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]

        print(f"检测到目标: {class_name}, 置信度: {confidence:.2f}, 边界框: {bbox}")

        # SAM 分割
        mask = sam_segment(img, bbox)

        # 保存每个单独目标的分割结果
        save_path = os.path.join(save_folder, f"segment_{idx}_{class_name}.jpg")
        save_individual_segment(img, mask, bbox, f"{class_name} {confidence:.2f}", save_path) 对代码的每一行进行分析



'''



import os
import torch
import cv2
import numpy as np
from pathlib import Path
from models.common import DetectMultiBackend
from segment_anything import sam_model_registry, SamPredictor
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
import matplotlib.pyplot as plt

# YOLOv9 目标检测模型路径
yolo_weights = r'E:\ABB\segment-anything\runs\train\exp19\weights\best.pt'
source = r'E:\ABB\AI\Depth-Anything-V2\Inter_Realsence_D345_Datasets\color_image_20241022-143158.jpg'

# SAM 模型加载路径
sam_checkpoint = r"E:\ABB\segment-anything\weights\sam_vit_h_4b8939.pth"
model_type = "vit_h"


# ----------------- YOLOv9 目标检测 -----------------

def yolo_detect(img_path):
    device = select_device('')
    model = DetectMultiBackend(yolo_weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt

    # 读取图片
    img0 = cv2.imread(img_path)
    img = cv2.resize(img0, (640, 480))  # 将图像调整为 YOLOv9 模型输入尺寸
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # 归一化
    img = img.permute(2, 0, 1).unsqueeze(0)  # 调整维度以适应模型输入

    # 目标检测推理
    pred = model(img)
    pred = non_max_suppression(pred, 0.25, 0.45)

    # 获取检测结果
    results = []
    for i, det in enumerate(pred):
        if len(det):
            # 将边界框调整回原图尺寸
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                results.append({
                    "bbox": [int(x) for x in xyxy],  # 边界框坐标
                    "confidence": float(conf),  # 置信度
                    "class": int(cls),  # 类别ID
                    "class_name": names[int(cls)]  # 类别名称
                })
    return img0, results


# ----------------- SAM 目标分割 -----------------

def sam_segment(img0, bbox):
    # 初始化 SAM 模型
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)

    # 转换图像颜色空间
    image = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    # 将 YOLO 边界框传递给 SAM 进行分割
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.array(bbox),
        multimask_output=False
    )

    return masks[0]  # 返回分割掩码


# ----------------- 单独保存每个分割目标 -----------------

def save_individual_segment(img0, mask, bbox, label, save_path):
    x0, y0, x1, y1 = bbox

    # 应用掩码到图像
    segmented_img = img0.copy()
    mask_area = mask > 0
    segmented_img[~mask_area] = 0  # 将非目标区域置为黑色

    # 裁剪目标区域
    cropped_img = segmented_img[y0:y1, x0:x1]

    # 保存裁剪的图像
    cv2.imwrite(save_path, cropped_img)
    print(f"目标 {label} 的分割图像已保存: {save_path}")


# ----------------- 保存带有所有目标分割掩码的完整图片 -----------------

def save_complete_segmented_image(img, masks, bboxes, labels, save_path):
    # 创建一张包含所有目标分割掩码的图像
    img_with_masks = img.copy()

    for mask, bbox, label in zip(masks, bboxes, labels):
        x0, y0, x1, y1 = bbox

        # 应用掩码
        mask_area = mask > 0

        # 使用红色高亮目标
        colored_mask = np.zeros_like(img_with_masks)
        colored_mask[mask_area] = [255, 0, 0]  # 用红色替代绿色

        # 对掩码进行模糊处理，使边缘更加平滑
        blurred_mask = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)

        # 将模糊后的掩码与原图进行混合
        img_with_masks[mask_area] = cv2.addWeighted(img_with_masks[mask_area], 0.7, colored_mask[mask_area], 0.3, 0)

        # 在图像上绘制目标检测边界框
        cv2.rectangle(img_with_masks, (x0, y0), (x1, y1), (0, 255, 0), 2)  # 使用绿色边框
        cv2.putText(img_with_masks, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # 保存带有所有分割结果的完整图片
    cv2.imwrite(save_path, img_with_masks)
    print(f"完整带有分割掩码的图像已保存: {save_path}")


# ----------------- 主函数 -----------------

if __name__ == "__main__":
    # YOLOv9 检测
    img, detections = yolo_detect(source)

    # 结果保存路径
    save_folder = r'E:\ABB\segment-anything\output_individual\SAM23'
    save_complete_image_path = r'/output_individual/SAM23/complete_segmented_image.jpg'
    os.makedirs(save_folder, exist_ok=True)

    # 遍历每个目标，使用 SAM 分割并保存结果
    masks = []
    bboxes = []
    labels = []

    # 遍历每个目标，使用 SAM 分割并保存结果
    for idx, detection in enumerate(detections):
        bbox = detection["bbox"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]

        print(f"检测到目标: {class_name}, 置信度: {confidence:.2f}, 边界框: {bbox}")

        # SAM 分割
        mask = sam_segment(img, bbox)

        # 保存每个单独目标的分割结果
        save_path = os.path.join(save_folder, f"segment_{idx}_{class_name}.jpg")
        save_individual_segment(img, mask, bbox, f"{class_name} {confidence:.2f}", save_path)

        # 保存信息用于完整图像保存
        masks.append(mask)
        bboxes.append(bbox)
        labels.append(f"{class_name} {confidence:.2f}")

    # 保存带有所有目标分割掩码的完整图片
    save_complete_segmented_image(img, masks, bboxes, labels, save_complete_image_path)
