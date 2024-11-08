import os
import torch
import cv2
import numpy as np
import pyrealsense2 as rs
from pathlib import Path
from models.common import DetectMultiBackend
from segment_anything import sam_model_registry, SamPredictor
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
import pandas as pd

# Paths
yolo_weights = r'E:\ABB\segment-anything\runs\train\exp19\weights\best.pt'
source = r'E:\ABB\AI\Depth-Anything-V2\Inter_Realsence_D345_Datasets\color_image_20241026-194402.jpg'
depth_image_path = r'E:\ABB\AI\Depth-Anything-V2\Inter_Realsence_D345_Datasets\depth_image_20241026-194402.png'
sam_checkpoint = r"E:\ABB\segment-anything\weights\sam_vit_h_4b8939.pth"
model_type = "vit_h"

# Transformation parameters for camera to simulation
alpha, beta, gamma = -148.0, -0.4, -178.0
tx, ty, tz = 0.525, 0.76, 1.25

# Initialize RealSense pipeline and get intrinsics
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Align depth to color
align = rs.align(rs.stream.color)

# Get intrinsics for depth camera
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
fx, fy, cx, cy = depth_intrinsics.fx, depth_intrinsics.fy, depth_intrinsics.ppx, depth_intrinsics.ppy
intrinsics = [fx, fy, cx, cy]


# YOLO Detection
def yolo_detect(img_path):
    device = select_device('')
    model = DetectMultiBackend(yolo_weights, device=device)
    img0 = cv2.imread(img_path)
    img = cv2.resize(img0, (640, 480))
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)
    pred = model(img)
    pred = non_max_suppression(pred, 0.25, 0.45)
    results = []
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                results.append({
                    "bbox": [int(x) for x in xyxy],
                    "confidence": float(conf),
                    "class": int(cls),
                    "class_name": model.names[int(cls)]
                })
    return img0, results


# SAM segmentation
def sam_segment(img0, bbox):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)
    predictor.set_image(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
    masks, _, _ = predictor.predict(point_coords=None, point_labels=None, box=np.array(bbox), multimask_output=False)
    return masks[0]


# Transformation matrix for camera to simulation
def get_transformation_matrix(alpha_deg, beta_deg, gamma_deg, tx, ty, tz):
    alpha, beta, gamma = np.radians([alpha_deg, beta_deg, gamma_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
    rotation_matrix = Rz @ Ry @ Rx
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = [tx, ty, tz]
    return transformation_matrix


# Extract point cloud from mask and convert to simulation coordinates
def extract_point_cloud_from_mask(depth_image, mask, intrinsics, transformation_matrix):
    points = []
    h, w = depth_image.shape
    fx, fy, cx, cy = intrinsics
    for y in range(h):
        for x in range(w):
            if mask[y, x]:
                depth = depth_image[y, x] * 0.001
                if depth > 0:
                    X = (x - cx) * depth / fx
                    Y = (y - cy) * depth / fy
                    Z = depth
                    camera_coords = np.array([X, Y, Z, 1])
                    world_coords = transformation_matrix @ camera_coords
                    points.append(world_coords[:3])
    return np.array(points)


if __name__ == "__main__":
    # Set up transformation matrix
    transformation_matrix = get_transformation_matrix(alpha, beta, gamma, tx, ty, tz)
    img0, detections = yolo_detect(source)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

    # Folders for results
    save_folder = r'E:\ABB\segment-anything\realsense_yolov9_simulation_detection\SAM27'
    os.makedirs(save_folder, exist_ok=True)

    complete_detection_img = img0.copy()
    complete_segmentation_img = img0.copy()

    # Process each detection
    for idx, detection in enumerate(detections):
        bbox = detection["bbox"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]

        # Segment with SAM
        mask = sam_segment(img0, bbox)

        # Extract and save point cloud
        point_cloud = extract_point_cloud_from_mask(depth_image, mask, intrinsics, transformation_matrix)
        save_path = os.path.join(save_folder, f"{class_name}_{idx}_point_cloud.npy")
        np.save(save_path, point_cloud)

        # Get the center point in simulation coordinates
        sim_coords = point_cloud.mean(axis=0).round(2)
        sim_coords_text = f"Sim XYZ: {sim_coords}"

        # Create individual annotated image with bounding box and simulation info
        x0, y0, x1, y1 = bbox
        annotated_img = img0.copy()
        cv2.rectangle(annotated_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.circle(annotated_img, (int((x0 + x1) / 2), int((y0 + y1) / 2)), 5, (255, 0, 0), -1)
        cv2.putText(annotated_img, f"{class_name} {confidence:.2f}", (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2)
        cv2.putText(annotated_img, sim_coords_text, (x0, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imwrite(os.path.join(save_folder, f"{class_name}_{idx}_annotated.jpg"), annotated_img)

        # Add info to complete detection and segmentation images
        cv2.rectangle(complete_detection_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.circle(complete_detection_img, (int((x0 + x1) / 2), int((y0 + y1) / 2)), 5, (255, 0, 0), -1)
        cv2.putText(complete_detection_img, f"{class_name} {confidence:.2f}", (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)
        cv2.putText(complete_detection_img, sim_coords_text, (x0, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                    2)

        # Apply mask to complete segmentation image
        mask_area = mask > 0
        colored_mask = np.zeros_like(complete_segmentation_img)
        colored_mask[mask_area] = [255, 0, 0]
        complete_segmentation_img[mask_area] = cv2.addWeighted(complete_segmentation_img[mask_area], 0.7,
                                                               colored_mask[mask_area], 0.3, 0)
        cv2.rectangle(complete_segmentation_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.circle(complete_segmentation_img, (int((x0 + x1) / 2), int((y0 + y1) / 2)), 5, (255, 0, 0), -1)
        cv2.putText(complete_segmentation_img, sim_coords_text, (x0, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 0), 2)

        # Save individual mask
        individual_img = img0.copy()
        individual_img[~mask_area] = 0
        cv2.imwrite(os.path.join(save_folder, f"{class_name}_{idx}_segmented.jpg"), individual_img)

        print(f"Class: {class_name}, Confidence: {confidence:.2f}, Simulation Position: {sim_coords_text}")

    # Save final combined images
    cv2.imwrite(os.path.join(save_folder, "complete_detection.jpg"), complete_detection_img)
    cv2.imwrite(os.path.join(save_folder, "complete_segmentation.jpg"), complete_segmentation_img)
