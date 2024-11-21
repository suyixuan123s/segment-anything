"""
Author: Yixuan Su
Date: 2024/11/21 12:08
File: Faster_R-CNN_SAM.py
Description:
"""

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import torchvision.transforms as T
import torchvision.models as models
import torchvision.ops as ops

# 导入预训练的模型（如 ResNet + Faster R-CNN 用于物体检测）
object_detection_model = models.detection.fasterrcnn_resnet50_fpn(weights='COCO_V1')
object_detection_model.eval()

# 加载 SAM 模型
model_path = r'/weights/sam_vit_h_4b8939.pth'
image_path = r'/data/images/00064.jpg'
sam = sam_model_registry["vit_h"](checkpoint=model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam.to(device)

# 加载图片
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 使用 SAM 进行全图分割
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)

# 准备输入对象检测模型
transform = T.Compose([T.ToTensor()])
image_tensor = transform(image_rgb).unsqueeze(0)

# 通过 Faster R-CNN 进行物体检测
with torch.no_grad():
    detections = object_detection_model(image_tensor)[0]

# 遍历分割掩码和检测结果
for i, mask in enumerate(masks):
    segmentation = mask['segmentation']
    mask_indices = np.where(segmentation)
    bbox = [min(mask_indices[1]), min(mask_indices[0]), max(mask_indices[1]), max(mask_indices[0])]

    # 将 SAM 生成的 bbox 转换为 Tensor
    sam_bbox = torch.tensor([bbox])

    # 计算 IoU 以匹配检测到的对象
    ious = ops.box_iou(sam_bbox, detections['boxes'])
    max_iou, max_idx = ious.max(dim=1)

    if max_iou.item() > 0.5:  # 只考虑 IoU 大于 0.5 的匹配
        label = detections['labels'][max_idx.item()].item()
        score = detections['scores'][max_idx.item()].item()
        print(f"Object {i}: Type {label}, Position {bbox}, Score {score}")


# 定义 show_anns 函数
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


# 显示图像和分割结果
plt.figure(figsize=(20, 20))
plt.imshow(image_rgb)
for mask in masks:
    show_anns([mask])
plt.axis('off')
plt.show()
