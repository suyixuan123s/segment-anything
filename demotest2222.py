# 导入必要的包
import os
import sys
import time
import argparse
import json

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

# 导入 segment_anything 包
from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# 输入必要的参数
model_path = r'./weights/sam_vit_h_4b8939.pth'
image_path = r'./demotest/00064.jpg'  # dog1 building1
output_folder = r'./demotest/building_prompt_results'
json_output_path = os.path.join(output_folder, "detection_info.json")

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 加载模型
sam = sam_model_registry["vit_h"](checkpoint=model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam.to(device)

# 输出模型加载完成的 current 时间
current_time1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print("Model loaded done", current_time1)
print("---------------------------------------------")

# 加载图片
image = cv2.imread(image_path)
# 输出图片加载完成的 current 时间
current_time2 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print("Image loaded done", current_time2)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# 这里是预测 不用提示词 进行全图分割
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)

# 输出预测完成的 current 时间
current_time3 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print("Predict loaded done", current_time3)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

# 假设每个掩码包含类型信息，例如 'type' 字段表示掩码的类型
# 在这个例子中，我们假设 'type' 字段已经在生成的掩码中定义
# 如果没有 'type' 字段，需要根据模型或实际情况调整代码以获取该信息

# 准备将信息写入 JSON 文件
detection_info = []
for i, mask in enumerate(masks):
    mask_type = mask.get('type', 'Unknown')  # 假设掩码中有 'type' 字段，若无则显示 'Unknown'
    mask_info = {
        "Mask Index": i + 1,
        "Area": mask['area'],
        "Bounding Box": mask['bbox'],
        "Segmentation Shape": mask['segmentation'].shape,
        "Type": mask_type,
        "Is Crowd": mask.get('iscrowd', 'N/A')
    }
    detection_info.append(mask_info)
    # 打印输出检测信息
    print(f"Mask {i + 1}:")
    print(f" - Type: {mask_info['Type']}")
    print(f" - Area: {mask_info['Area']}")
    print(f" - Bounding Box: {mask_info['Bounding Box']}")
    print(f" - Segmentation Shape: {mask_info['Segmentation Shape']}")
    print(f" - Is Crowd: {mask_info['Is Crowd']}")
    print("")

# 将检测信息写入 JSON 文件
with open(json_output_path, 'w') as json_file:
    json.dump(detection_info, json_file, indent=4)

print(f"Detection information saved to {json_output_path}")

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

# 展示预测结果 img 和 mask
plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()
