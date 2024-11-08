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
    img = cv2.resize(img0, (640, 640))  # 将图像调整为 YOLOv9 模型输入尺寸
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


# ----------------- 显示结果 -----------------

def show_result(img, masks, bbox, label):
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    plt.imshow(img)

    # 显示分割掩码
    mask = masks.astype(np.uint8)
    colored_mask = np.zeros_like(img)
    colored_mask[mask > 0] = [0, 255, 0]  # 用绿色高亮目标
    plt.imshow(colored_mask, alpha=0.5)

    # 显示目标检测边界框
    x0, y0, x1, y1 = bbox
    rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='red', facecolor='none', lw=2)
    ax.add_patch(rect)

    # 显示目标类别标签
    ax.text(x0, y0 - 10, label, color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
    plt.axis('off')
    plt.show()


# ----------------- 主函数 -----------------

if __name__ == "__main__":
    # YOLOv9 检测
    img, detections = yolo_detect(source)

    # 遍历每个目标，使用 SAM 分割并显示结果
    for detection in detections:
        bbox = detection["bbox"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]

        print(f"检测到目标: {class_name}, 置信度: {confidence:.2f}, 边界框: {bbox}")

        # SAM 分割
        mask = sam_segment(img, bbox)

        # 显示检测和分割结果
        show_result(img, mask, bbox, f"{class_name} {confidence:.2f}")
