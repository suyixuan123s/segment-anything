'''
Description:
Author: tianyw
Date: 2024-03-01 14:06:05
LastEditTime: 2024-03-01 15:46:54
LastEditors: tianyw
'''



# 导入必要的包
import os
import time
import json

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# 导入 segment_anything 包
from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator

# 输入必要的参数
# 模型路径
model_path = r'.\weights/sam_vit_h_4b8939.pth'
# 输入的图片文件夹路径
image_folder = r'E:\ABB\segment-anything\image_capture\round3'
# 输出的文件夹路径
output_folder = r'E:\ABB\segment-anything\image_capture\building_prompt_results\round3'

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

def process_image(image_path):
    # 加载图片
    image = cv2.imread(image_path)
    # 输出图片加载完成的 current 时间
    current_time2 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("Image loaded done", current_time2)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # 这里是预测 不用提示词 进行全图分割
    mask_generator = SamAutomaticMaskGenerator(sam)

    # 开始计时
    start_time = time.time()

    # 模型预测
    masks = mask_generator.generate(image)

    # 结束计时
    end_time = time.time()

    # 计算处理时间
    processing_time = end_time - start_time

    # 输出预测完成的 current 时间
    current_time3 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("Predict loaded done", current_time3)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # 转换 masks 中的 ndarray 为列表
    def convert_masks_to_serializable(masks):
        serializable_masks = []
        for mask in masks:
            serializable_mask = {
                'segmentation': mask['segmentation'].tolist(),
                'area': mask['area'],
                'bbox': mask['bbox'],
                'predicted_iou': mask['predicted_iou'],
                'point_coords': mask['point_coords'],
                'stability_score': mask['stability_score'],
                'crop_box': mask['crop_box']
            }
            serializable_masks.append(serializable_mask)
        return serializable_masks

    # 保存分割信息到JSON文件中
    output_json_path = os.path.join(output_folder, os.path.basename(image_path).split('.')[0] + '.json')

    segmentation_info = {
        'image_path': image_path,
        'model_path': model_path,
        'time_info': {
            'model_loaded_time': current_time1,
            'image_loaded_time': current_time2,
            'predict_done_time': current_time3,
            'processing_time_seconds': processing_time
        },
        'masks': convert_masks_to_serializable(masks)  # 将 masks 转换为可序列化的形式
    }

    with open(output_json_path, 'w') as json_file:
        json.dump(segmentation_info, json_file, indent=4)

    print(f'Segmentation information saved to {output_json_path}')

    # 保存分割后的图像
    def save_annotated_image(image, masks, output_image_path):
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

        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        show_anns(masks)
        plt.axis('off')
        plt.savefig(output_image_path, bbox_inches='tight')
        plt.close()

    output_image_path = os.path.join(output_folder, os.path.basename(image_path))
    save_annotated_image(image, masks, output_image_path)

    print(f'Annotated image saved to {output_image_path}')

# 处理文件夹中的所有图片
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(image_folder, filename)
        process_image(image_path)
