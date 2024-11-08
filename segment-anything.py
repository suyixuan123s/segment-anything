import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2


# 添加掩码
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns[10:200]:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


image = cv2.imread('./images/05.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image.shape)

plt.figure(figsize=(20, 20))
plt.imshow(image)
plt.axis('off')
plt.show()

import sys

sys.path.append("..")
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "../weights/mobile_sam.pt"
model_type = "vit_t"

device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
sam.eval()

mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(image)

plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks[:])
plt.axis('off')
plt.show()


# 保存掩码
def save_mask(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=False)
    img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 1))

    for ann in sorted_anns[:]:
        m = ann['segmentation']
        img[m] = 255

    cv2.imwrite('res.jpg', img)


# save_mask(masks)
sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
save_mask(sorted_anns[:])


# 获取边缘
import cv2
import numpy as np

image = cv2.imread('./images/05.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
img_ = np.zeros_like(image)
gray_images = mask_show(masks[:])
for img in gray_images[:]:
    gray_image = np.uint8(img)
    gray_image = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
    edges = cv2.Canny(gray_image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_, contours, -1, (255, 255, 255), 2)
cv2.imwrite("counte2.png", img_)
# 蒙版-边缘
im = cv2.imread('images/05.jpg', cv2.IMREAD_GRAYSCALE)
image1 = cv2.imread('res.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('counte2.png', cv2.IMREAD_GRAYSCALE)
img = cv2.subtract(image1, image2)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
dst2 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# print(dst2.shape)

re_img = cv2.addWeighted(dst2, 0.2, im, 0.8, 0)
cv2.imwrite("res3.jpg", dst2)

plt.figure(figsize=(20, 20))
plt.imshow(dst2, cmap='gray')
plt.axis('off')
plt.show()

# 以COCO格式存储
import json

orig_img = cv2.imread('./images/05.jpg')
image = cv2.imread('res3.jpg')
edges = cv2.Canny(image, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
images = [
    {
        'file_name': '05.jpg',
        'height': int(orig_img.shape[0]),
        'width': int(orig_img.shape[1]),
        'id': 1
    },
]

categories = [
    {
        'id': 1,
        'name': 'qituan'
    },
]
annotations = []
for contour in contours:
    seg = []
    contour_ = contour.squeeze(1)
    seg.append(list(contour_.flatten().tolist()))
    x, y, w, h = cv2.boundingRect(contour)
    bbox = [x, y, w, h]
    area = cv2.contourArea(contour)
    iscrowd = 0
    image_id = 1
    category_id = 1
    id = len(annotations) + 1
    annotations.append({
        'segmentation': seg,
        'bbox': bbox,
        'area': area,
        'iscrowd': 0,
        'image_id': 1,
        'category_id': 1,
        'id': id
    })

coco_data = {
    'images': images,
    'annotations': annotations,
    'categories': categories
}

print(coco_data)

output_file_path = 'coco_data.json'

# Serialize the data and write to a JSON file
with open(output_file_path, 'w') as f:
    json.dump(coco_data, f, indent=4)
