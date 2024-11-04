import cv2
import numpy as np
import random


def read_points_from_file(file_path):
    pixel_points = []
    world_points = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("Pixel") and "Point" in line:
                # 提取像素坐标
                pixel_coords = line.split(':')[0].replace("Pixel", "").replace("(", "").replace(")", "").strip()
                x, y = map(int, pixel_coords.split(","))
                pixel_points.append((x, y))

                # 提取3D坐标
                point_coords = line.split(':')[1].replace("Point", "").replace("[", "").replace("]", "").strip()
                px, py, pz = map(float, point_coords.split())
                world_points.append((px, py, pz))
    return np.array(pixel_points, dtype=np.float32), np.array(world_points, dtype=np.float32)


# PnP和RANSAC参数
def ransac_pnp(pixel_points, world_points, camera_matrix, dist_coeffs, threshold=1.4, max_iterations=1000):
    max_inliers = 0
    best_rvec = None
    best_tvec = None

    for _ in range(max_iterations):
        # 随机选择3对点
        indices = random.sample(range(len(pixel_points)), 3)
        sampled_pixel_points = pixel_points[indices]
        sampled_world_points = world_points[indices]

        # 使用PnP求解位姿
        success, rvec, tvec = cv2.solvePnP(sampled_world_points, sampled_pixel_points, camera_matrix, dist_coeffs)

        if success:
            # 计算重投影误差，判断内点
            inliers_count = 0
            for i in range(len(pixel_points)):
                projected_point, _ = cv2.projectPoints(world_points[i].reshape(-1, 3), rvec, tvec, camera_matrix,
                                                       dist_coeffs)
                projected_point = projected_point.reshape(-1, 2)

                # 计算重投影误差
                error = np.linalg.norm(pixel_points[i] - projected_point)
                if error < threshold:
                    inliers_count += 1

            # 更新最佳位姿
            if inliers_count > max_inliers:
                max_inliers = inliers_count
                best_rvec, best_tvec = rvec, tvec

    return best_rvec, best_tvec, max_inliers


# 主函数
if __name__ == "__main__":
    # 文件路径
    file_path = "temp/match_point_pair.txt"


    pixel_points, world_points = read_points_from_file(file_path)

    # 定义相机内参矩阵（根据你的相机内参设置）
    # 例如：fx, fy, cx, cy
    fx, fy = 3438.924, 3407.551
    cx, cy = 1755.586, 2313.788
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros(5)  # 假设无畸变

    # 使用RANSAC和PnP估算位姿
    best_rvec, best_tvec, max_inliers = ransac_pnp(pixel_points, world_points, camera_matrix, dist_coeffs)

    # 输出结果
    if best_rvec is not None and best_tvec is not None:
        print("最佳旋转向量 (rvec):\n", best_rvec)
        print("最佳平移向量 (tvec):\n", best_tvec)
        print("最大内点数量:", max_inliers)
    else:
        print("未找到合适的位姿解。")
