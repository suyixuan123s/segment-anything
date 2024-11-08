import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from Test.YOLOv9_Detect_API import DetectAPI  # 假设你已经定义了YOLOv9的API

# 类别名称字典，将编号映射为具体的类别名称
class_names = {
    0: 'blood_tube',
    1: '5ML_centrifuge_tube',
    2: '10ML_centrifuge_tube',
    3: '5ML_sorting_tube_rack',
    4: '10ML_sorting_tube_rack',
    5: 'centrifuge_open',
    6: 'centrifuge_close',
    7: 'refrigerator_open',
    8: 'refrigerator_close',
    9: 'operating_desktop',
    10: 'tobe_sorted_tube_rack',
    11: 'dispensing_tube_rack',
    12: 'sorting_tube_rack_base',
    13: 'tube_rack_storage_cabinet'
}

# 加载 YOLOv9 模型
model = DetectAPI(weights='E:/ABB/AI/yolov9/runs/train/exp19/weights/best.pt')  # 替换为你的YOLOv9模型路径

# 初始化 SAM 模型
sam_checkpoint = r"E:\ABB\segment-anything\weights\sam_vit_h_4b8939.pth"  # SAM 模型权重路径
model_type = "vit_h"  # 使用的 SAM 模型类型，例如 "vit_b", "vit_l", "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

# 加载待处理的图像
image_path = r"E:\ABB\segment-anything\demotest\00048.jpg"  # 替换为你的图像路径
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 获取 YOLOv9 检测结果，注意这里传入的是图像列表
detections = model.detect([image])  # 假设 `detect` 方法现在接收的是图像列表，并返回 [x_min, y_min, x_max, y_max, confidence, class] 格式的结果

# 打印检测结果以调试
print("Detections:", detections)

# 遍历每个检测框，并使用 SAM 进行分割
masks = []
for detection in detections:
    # 打印每个 detection 的内容，检查其格式
    print("Detection:", detection)

    # 获取检测框的坐标和类别编号
    try:
        x_min, y_min, x_max, y_max = map(int, detection[:4])  # 获取检测框坐标
        confidence = detection[4]  # 置信度
        class_id = int(detection[5])  # 类别编号
        class_name = class_names.get(class_id, "Unknown")  # 根据类别编号获取类别名称
    except TypeError as e:
        print(f"Error converting detection to int: {e}")
        continue  # 跳过该检测框

    print(f"Detected {class_name} with confidence {confidence:.2f} at [{x_min}, {y_min}, {x_max}, {y_max}]")

    # 使用 SAM 预测
    predictor.set_image(image_rgb)

    # SAM 需要一个输入框来进行预测，这里可以使用检测框的坐标
    input_box = np.array([[x_min, y_min, x_max, y_max]])
    masks_pred, scores, logits = predictor.predict(box=input_box, multimask_output=True)

    # 选择最高得分的掩码
    mask = masks_pred[np.argmax(scores)]
    masks.append(mask)

    # 如果需要在原图上显示掩码，可以这样绘制
    mask_image = np.zeros_like(image_rgb)
    mask_image[mask] = (0, 255, 0)  # 用绿色显示掩码区域
    image_with_mask = cv2.addWeighted(image_rgb, 0.7, mask_image, 0.3, 0)

    # 保存分割后的图像（可选）
    output_path = f"output_segmented_{class_name}_{x_min}_{y_min}.png"
    cv2.imwrite(output_path, cv2.cvtColor(image_with_mask, cv2.COLOR_RGB2BGR))
    print(f"Saved segmented image as {output_path}")

print("Segmentation and mask extraction completed.")
