import os
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from suyixuan.Task1_YOLOv9_Detect.YOLOv9_Detect_API import DetectAPI

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
image_path = r"/Dataset_sets/demotest/00048.jpg"  # 替换为你的图像路径
color_image = cv2.imread(image_path)

# 检查图像是否加载成功
if color_image is None:
    print(f"Failed to load image from {image_path}")
else:
    # 将彩色图像作为YOLOv9模型的输入
    source = [color_image]

    # YOLOv9模型预测
    im0, pred = model.detect(source)

    # 创建用于叠加掩码的图层
    mask_layer = np.zeros_like(color_image, dtype=np.uint8)

    if pred is not None and len(pred):
        for det in pred:
            if len(det):
                for *xyxy, conf, cls in det:
                    x_min, y_min, x_max, y_max = map(int, xyxy)
                    class_id = int(cls)
                    class_name = class_names.get(class_id, "Unknown")

                    # 使用 SAM 对检测框内的区域进行分割
                    predictor.set_image(color_image)
                    input_box = np.array([[x_min, y_min, x_max, y_max]])
                    masks_pred, scores, _ = predictor.predict(box=input_box, multimask_output=True)

                    # 选择得分最高的掩码
                    best_mask = masks_pred[np.argmax(scores)]

                    # 将掩码叠加到 mask_layer 上，设置为不同的颜色，例如使用绿色
                    mask_layer[best_mask] = (0, 255, 0)  # 使用绿色高亮

                    # 在图像上绘制检测框
                    cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(color_image, class_name, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 将原始图像与掩码层叠加
    highlighted_image = cv2.addWeighted(color_image, 0.7, mask_layer, 0.3, 0)

    # 保存或显示高亮结果
    output_path = r"/Dataset_sets/demotest/highlighted_output.png"
    cv2.imwrite(output_path, highlighted_image)
    print(f"Highlighted image saved to {output_path}")

    # 显示结果
    cv2.imshow('Segmentation Highlight', highlighted_image)
    cv2.waitKey(0)  # 按任意键关闭窗口
    cv2.destroyAllWindows()
