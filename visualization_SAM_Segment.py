import yaml
from YOLOv9.capture_detect_results_confidence import class_ids, bboxes, conf_scores
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ----------------- 初始化 SAM -----------------
# 加载预训练模型
sam_checkpoint = r"E:\ABB\segment-anything\weights\sam_vit_h_4b8939.pth"  # 替换为实际路径
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

# ----------------- 加载图像并设置图像 -----------------
image_path = r'E:\ABB\AI\Depth-Anything-V2\Inter_Realsence_D345_Datasets\color_image_20241022-143158.jpg'  # 替换为实际路径
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

# ----------------- 显示图像和分割结果 -----------------
plt.figure(figsize=(10, 10))
ax = plt.gca()
plt.imshow(image)

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

    # 显示掩码和边界框
    show_mask(masks[0], ax, color=color)
    show_box(bbox, class_name, conf, color, ax)

# 显示最终的图像
plt.axis('off')
plt.show()
