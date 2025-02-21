import json
import numpy as np
import cv2
import open3d as o3d

# Load camera parameters from JSON file
with open("intrinsics.json", "r") as f:
    params = json.load(f)

color_intr = params["color_intrinsics"]
depth_intr = params["depth_intrinsics"]

color_camera_matrix = np.array([
    [color_intr["fx"], 0, color_intr["ppx"]],
    [0, color_intr["fy"], color_intr["ppy"]],
    [0, 0, 1]
])

depth_camera_matrix = np.array([
    [depth_intr["fx"], 0, depth_intr["ppx"]],
    [0, depth_intr["fy"], depth_intr["ppy"]],
    [0, 0, 1]
])

# Load images
color_image = cv2.imread("rgb_frame/color_2.png")
depth_image = cv2.imread("depth_frame/depth_2.png", cv2.IMREAD_UNCHANGED).astype(np.float32)

# Convert depth image to meters
depth_image_meters = depth_image / 1000.0

# Get pixel coordinates
height, width = depth_image.shape
u, v = np.meshgrid(np.arange(width), np.arange(height))
u = u.flatten()
v = v.flatten()
z = depth_image_meters.flatten()

# Filter valid depth points
valid = z > 0
u = u[valid]
v = v[valid]
z = z[valid]

# Convert depth pixels to 3D points (depth camera coordinates)
x = (u - depth_intr["ppx"]) * z / depth_intr["fx"]
y = (v - depth_intr["ppy"]) * z / depth_intr["fy"]
depth_points = np.vstack((x, y, z))

# Project 3D points onto the color image plane
uv_homog = color_camera_matrix @ depth_points
u_color = (uv_homog[0] / uv_homog[2]).astype(int)
v_color = (uv_homog[1] / uv_homog[2]).astype(int)

# Filter points inside the color image bounds
valid = (u_color >= 0) & (u_color < color_intr["width"]) & \
        (v_color >= 0) & (v_color < color_intr["height"])

u_color = u_color[valid]
v_color = v_color[valid]
z = z[valid]

# Create aligned depth image
aligned_depth_image = np.zeros((color_intr["height"], color_intr["width"]), dtype=np.float32)
aligned_depth_image[v_color, u_color] = z

# Normalize and convert depth images to color maps
original_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=255.0 / np.max(depth_image)), cv2.COLORMAP_JET)
aligned_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(aligned_depth_image, alpha=255.0 / np.max(aligned_depth_image)), cv2.COLORMAP_JET)

# Display images
cv2.imshow("Side-by-Side Comparison: Original", np.hstack((color_image, original_depth_colormap)))
cv2.imshow("Side-by-Side Comparison: Aligned", np.hstack((color_image, aligned_depth_colormap)))

# Generate 3D Point Cloud
valid_points = depth_points.T[valid]
valid_colors = color_image[v_color, u_color][:, ::-1] / 255.0

# Convert to Open3D format
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(valid_points)
pcd.colors = o3d.utility.Vector3dVector(valid_colors)

o3d.visualization.draw_geometries([pcd], window_name="3D Point Cloud", width=800, height=600)

cv2.waitKey(1)
cv2.destroyAllWindows()
