import open3d as o3d
import numpy as np
import cv2

# 加载点云
pcd = o3d.io.read_point_cloud(r'data/point_cloud_cluster/3.pcd')

# 1. 使用RANSAC平面拟合
plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)

[a, b, c, d] = plane_model  # 平面方程 ax + by + cz + d = 0
print(f"拟合平面方程: {a}x + {b}y + {c}z + {d} = 0")

# 提取平面内的点云（拟合平面的内点）
inlier_cloud = pcd.select_by_index(inliers)
inlier_points = np.asarray(pcd.points)
inlier_colors = np.asarray(pcd.colors)

# 2. 计算旋转矩阵，将平面法向量 (a, b, c) 对齐到 Z 轴 (0, 0, 1)
plane_normal = np.array([a, b, c])
plane_normal_normalized = plane_normal / np.linalg.norm(plane_normal)
z_axis = np.array([0, 0, 1])

# 计算旋转轴和角度
v = np.cross(plane_normal_normalized, z_axis)  # 旋转轴
s = np.linalg.norm(v)  # sin(theta)
c_angle = np.dot(plane_normal_normalized, z_axis)  # cos(theta)
I = np.eye(3)

if s != 0:
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])  # 反对称矩阵
    R = I + vx + np.matmul(vx, vx) * ((1 - c_angle) / (s ** 2))  # Rodrigues旋转公式
else:
    R = I  # 如果平面法向量已经与Z轴对齐，无需旋转

# 旋转点云，使得拟合平面与XY平面对齐
rotated_points = inlier_points @ R.T

# 3. 平移平面，确保所有点位于平面的上方（Z >= 0）
dists_to_plane = rotated_points[:, 2]
min_dist = np.min(dists_to_plane)
rotated_points[:, 2] -= min_dist

# 4. 将点云投影到XY平面上
projected_points = rotated_points.copy()
projected_points[:, 2] = 0

# 5. 确定图像尺寸并将投影后的点映射到图像坐标
min_x, min_y = np.min(projected_points[:, :2], axis=0)
max_x, max_y = np.max(projected_points[:, :2], axis=0)

img_width = int((max_x - min_x) * 50)  # 调整因子 50 来增加图像的分辨率
img_height = int((max_y - min_y) * 50)

scale_x = (img_width - 1) / (max_x - min_x) if max_x != min_x else 1.0
scale_y = (img_height - 1) / (max_y - min_y) if max_y != min_y else 1.0
scale = min(scale_x, scale_y)

offset_x = -min_x * scale
offset_y = -min_y * scale

# 创建RGB图像，初始化为黑色
image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
depth_map = np.ones((img_height, img_width), dtype=np.float32) * np.inf
depth_image = np.ones((img_height, img_width), dtype=np.uint8) * 255

# 创建一个字典，用于存储图像像素坐标与原始点云坐标的映射
pixel_to_point_map = {}

# 6. 遍历每个投影点，将其颜色映射到对应的像素位置
for i, point in enumerate(projected_points):
    u = int(point[0] * scale + offset_x)
    v = int(point[1] * scale + offset_y)
    if 0 <= u < img_width and 0 <= v < img_height:
        dist_to_plane = rotated_points[i, 2]

        if dist_to_plane < depth_map[v, u]:
            depth_map[v, u] = dist_to_plane
            color = (inlier_colors[i] * 255).astype(np.uint8)
            image[img_height - v - 1, u] = color

            # 将深度映射到0-255的范围，生成深度图像
            depth_value = int(255 * (dist_to_plane / (np.max(depth_map) - np.min(depth_map))))
            depth_image[img_height - v - 1, u] = depth_value

            # 记录图像坐标 (u, v) 对应的原始点云坐标
            pixel_to_point_map[(u, v)] = inlier_points[i]

# 7. 保存映射关系到文件
np.save('temp/pixel_to_point_map.npy', pixel_to_point_map)
# 或者保存为文本文件
with open('temp/pixel_to_point_map.txt', 'w') as f:
    for (u, v), point in pixel_to_point_map.items():
        f.write(f"Pixel ({u}, {v}): Point {point}\n")

# 保存并显示RGB图像
cv2.imwrite('temp/projected_image_fixed.png', image)
cv2.imwrite('temp/depth_image.png', depth_image)

cv2.imshow("Projected Image", image)
cv2.imshow("Depth Image", depth_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("投影图像和深度图已保存为: projected_image_fixed.png 和 depth_image.png")
print("像素到点云映射已保存为 pixel_to_point_map.npy 和 pixel_to_point_map.txt")
