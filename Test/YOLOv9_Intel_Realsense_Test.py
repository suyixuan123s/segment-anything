import time
import cv2
import torch
import numpy as np
import pyrealsense2 as rs
import YOLOv9_Detect_API

pipeline = rs.pipeline()  # 定义流程pipeline，创建一个管道
config = rs.config()  # 定义配置config

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 配置depth流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 配置color流

pipe_profile = pipeline.start(config)  # streaming流开始
align = rs.align(rs.stream.color)


def get_aligned_images():
    frames = pipeline.wait_for_frames()  # 等待获取图像帧，获取颜色和深度的框架集
    aligned_frames = align.process(frames)  # 获取对齐帧，将深度框与颜色框对齐

    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的的depth帧
    aligned_color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的的color帧

    #### 获取相机参数 ####
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
    color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参

    img_color = np.asanyarray(aligned_color_frame.get_data())  # RGB图
    img_depth = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）

    depth_colormap = cv2.applyColorMap \
        (cv2.convertScaleAbs(img_depth, alpha=0.008)
         , cv2.COLORMAP_JET)

    return color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame


''' 
获取随机点三维坐标
'''

def get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin):
    x = depth_pixel[0]
    y = depth_pixel[1]
    dis = aligned_depth_frame.get_distance(x, y)  # 获取该像素点对应的深度
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
    return dis, camera_coordinate


if __name__ == '__main__':  # 入口

    a = YOLOv9_Detect_API.DetectAPI(weights='E:/ABB/AI/yolov9/runs/train/exp19/weights/best.pt')
    # 设置计时器
    start_time = time.time()
    interval = 1  # 间隔时间（秒）
    try:
        while True:
            color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame = get_aligned_images()  # 获取对齐图像与相机参数
            if not img_color.any() or not img_depth.any():
                continue
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
                img_depth, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((img_color, depth_colormap))

            # 检查是否达到间隔时间
            if time.time() - start_time >= interval:

                start_time = time.time()  # 重置计时器

                # 调用 detect 函数并获取预测结果
                im0, pred = a.detect([img_color])

                # 初始化空列表来存储结果
                camera_xyz_list = []
                class_id_list = []
                xyxy_list = []
                conf_list = []

                # 检查是否有预测结果
                if pred is not None and len(pred):
                    for det in pred:  # 处理每个检测结果
                        if len(det):
                            for *xyxy, conf, cls in det:
                                xyxy_list.append(xyxy)
                                class_id_list.append(int(cls))
                                conf_list.append(float(conf))

                                # 获取目标中心点的像素坐标并计算3D坐标
                                ux = int((xyxy[0] + xyxy[2]) / 2)  # 计算x中心
                                uy = int((xyxy[1] + xyxy[3]) / 2)  # 计算y中心
                                dis = aligned_depth_frame.get_distance(ux, uy)
                                camera_xyz = rs.rs2_deproject_pixel_to_point(
                                    depth_intrin, (ux, uy), dis)  # 计算相机坐标系的xyz
                                camera_xyz = np.round(np.array(camera_xyz), 3)  # 转成3位小数
                                camera_xyz = np.array(list(camera_xyz)) * 1000  # 单位转换为毫米
                                camera_xyz = list(camera_xyz)

                                # 在图像上绘制中心点和3D坐标
                                cv2.circle(im0, (ux, uy), 4, (255, 255, 255), 5)  # 标出中心点
                                cv2.putText(im0, str(camera_xyz), (ux + 20, uy + 10), 0, 1,
                                            [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)  # 标出坐标
                                camera_xyz_list.append(camera_xyz)

                # 显示检测结果
                cv2.namedWindow('detection', flags=cv2.WINDOW_NORMAL |
                                                   cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
                cv2.resizeWindow('detection', 640, 480)
                cv2.imshow('detection', im0)
                cv2.waitKey(2000)

            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                pipeline.stop()
                break
    finally:
        # Stop streaming
        pipeline.stop()