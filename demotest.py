'''
Description:
Author: tianyw
Date: 2024-03-01 14:06:05
LastEditTime: 2024-03-01 15:46:54
LastEditors: tianyw
'''

# https://blog.csdn.net/yinweimumu/article/details/136432874?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1-136432874-blog-130601007.235^v43^pc_blog_bottom_relevance_base7&spm=1001.2101.3001.4242.2&utm_relevant_index=4

# 导入必要的包
import os
import sys
import time
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

# 导入 segment_anything 包
from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything import sam_model_registry

# 输入必要的参数
# E:\1Study\01AI\01segment\segment-anything\codes\segment-anything
# 模型路径
model_path = r'./weights/sam_vit_h_4b8939.pth'
# 输入的图片路径
image_path = r'./demotest/00064.jpg'  # dog1 building1
# 输出的图片路径
output_folder = r'./demotest/building_prompt_results'

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

# 使用提示词，进行局部分割
# predictor = SamPredictor(sam)
# predictor.set_image(image)
# masks,scores,logits = predictor.predict(point_coords=None,point_labels=None,box=None,mask_input=None,multimask_output=True,return_logits=True)


# 输出预测完成的 current 时间
current_time3 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print("Predict loaded done", current_time3)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


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


