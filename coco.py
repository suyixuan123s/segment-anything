import cv2
import random
import json, os
from pycocotools.coco import COCO
from skimage import io
from matplotlib import pyplot as plt
import numpy as np

train_json = 'coco_data.json'
train_path = './images/'
coco = COCO(train_json)

list_imgIds = coco.getImgIds(catIds=1)
list_imgIds

img = coco.loadImgs(list_imgIds[0])[0]
image = cv2.imread(train_path + img['file_name'])  # 读取图像
img_annIds = coco.getAnnIds(imgIds=1, catIds=1, iscrowd=None)
anns = coco.loadAnns(img_annIds)
img = coco.loadImgs(list_imgIds[0])[0]
img1 = cv2.imread(train_path + img['file_name'])  # 读取图像
# 分割
for ann in anns:
    data = np.array(ann['segmentation'][0])
    num_points = len(data) // 2
    contour_restored = data.reshape((num_points, 2))
    contour_restored = contour_restored.reshape(contour_restored.shape[0], 1, contour_restored.shape[1])
    # print(contour_restored.shape)
    color = np.random.randint(0, 255, 3).tolist()
    cv2.drawContours(img1, [contour_restored], -1, tuple(color), thickness=cv2.FILLED)

    # mask = coco.annToMask(ann)
    # color = np.random.randint(0, 255, 3)  # Random color for each mask
    # img = cv2.addWeighted(img, 1, cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR), 0.5, 0)

plt.rcParams['figure.figsize'] = (20.0, 20.0)
# 此处的20.0是由于我的图片是2000*2000，目前还没去研究怎么利用plt自动分辨率。
plt.imshow(img1)
plt.show()

img_annIds = coco.getAnnIds(imgIds=1, catIds=1, iscrowd=None)
img_annIds
# 目标检测
for id in img_annIds[:]:
    ann = coco.loadAnns(id)[0]
    x, y, w, h = ann['bbox']
    # print(ann['bbox'])
    image1 = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2)

plt.rcParams['figure.figsize'] = (20.0, 20.0)
# 此处的20.0是由于我的图片是2000*2000，目前还没去研究怎么利用plt自动分辨率。
plt.imshow(image1)
plt.show()