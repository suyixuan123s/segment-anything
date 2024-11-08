import open3d as o3d

# 读取点云文件
point_cloud_path = r'E:\ABB\segment-anything\point_cloud_data\blood_tube_point_cloud.ply'
pcd = o3d.io.read_point_cloud(point_cloud_path)

# 检查是否成功读取点云
if pcd.is_empty():
    print("点云文件为空或无法读取！")
else:
    # 打印点云信息
    print("点云信息:")
    print(pcd)

    # 可视化点云
    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Visualization",
                                      width=800, height=600, left=50, top=50,
                                      point_show_normal=False)
