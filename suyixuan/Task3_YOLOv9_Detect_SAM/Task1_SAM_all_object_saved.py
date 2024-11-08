import os
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

# SAM 模型加载路径
sam_checkpoint = r"E:\ABB\segment-anything\weights\sam_vit_h_4b8939.pth"
model_type = "vit_h"


# ----------------- 读取 YOLOv9 的检测框数据 -----------------

def load_yolo_boxes(yolo_txt_path, img_width, img_height):
    boxes = []
    labels = []
    with open(yolo_txt_path, 'r') as f:
        for line in f.readlines():
            data = line.strip().split()
            class_id = int(data[0])  # 类别 ID
            x_center = float(data[1]) * img_width
            y_center = float(data[2]) * img_height
            width = float(data[3]) * img_width
            height = float(data[4]) * img_height
            confidence = float(data[5])  # 置信度

            x0 = int(x_center - width / 2)
            y0 = int(y_center - height / 2)
            x1 = int(x_center + width / 2)
            y1 = int(y_center + height / 2)

            boxes.append([x0, y0, x1, y1])
            labels.append(f"Class {class_id} {confidence:.2f}")
    return boxes, labels


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


# ----------------- 保存带有所有目标分割掩码的完整图片 -----------------

def save_complete_segmented_image(img, masks, bboxes, labels, save_path):
    # 创建一张包含所有目标分割掩码的图像
    img_with_masks = img.copy()

    for mask, bbox, label in zip(masks, bboxes, labels):
        x0, y0, x1, y1 = bbox

        # 应用掩码
        mask_area = mask > 0
        colored_mask = np.zeros_like(img_with_masks)
        colored_mask[mask_area] = [0, 255, 0]  # 用绿色高亮目标

        # 将掩码叠加到原图
        img_with_masks[mask_area] = cv2.addWeighted(img_with_masks[mask_area], 0.5, colored_mask[mask_area], 0.5, 0)

        # 在图像上绘制目标检测边界框
        cv2.rectangle(img_with_masks, (x0, y0), (x1, y1), (0, 0, 255), 2)  # 红色边框
        cv2.putText(img_with_masks, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # 保存带有所有分割结果的完整图片
    cv2.imwrite(save_path, img_with_masks)
    print(f"完整带有分割掩码的图像已保存: {save_path}")


# ----------------- 主函数 -----------------

if __name__ == "__main__":
    # YOLOv9 检测输出的图片路径和检测框文件
    yolo_txt_path = '../Task1_YOLOv9_Detect/detect/exp2/color_image_20241026-194402.txt'
    image_path = r'color_image_20241026-194402.jpg'

    # 结果保存路径
    save_folder_individual = r'SAM'
    save_complete_image_path = r'complete_segmented_image.jpg'
    os.makedirs(save_folder_individual, exist_ok=True)

    # 加载带有检测框的图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法加载图片: {image_path}")
        exit()

    # 获取图片尺寸
    img_height, img_width = img.shape[:2]

    # 从 YOLO txt 文件中加载边界框和标签
    bboxes, labels = load_yolo_boxes(yolo_txt_path, img_width, img_height)

    masks = []  # 用于保存所有分割掩码

    # 遍历每个目标，使用 SAM 分割并保存单独的分割结果
    for idx, bbox in enumerate(bboxes):
        label = labels[idx]

        # SAM 分割
        mask = sam_segment(img, bbox)
        masks.append(mask)

        # 保存每个单独目标的分割结果
        save_path = os.path.join(save_folder_individual, f"segment_{idx}_{label}.jpg")
        save_individual_segment(img, mask, bbox, label, save_path)

    # 保存带有所有目标分割掩码的完整图片

    save_complete_segmented_image(img, masks, bboxes, labels, save_complete_image_path)
