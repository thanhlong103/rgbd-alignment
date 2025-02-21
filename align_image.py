import json
import numpy as np
import cv2
import open3d as o3d

# Load parameters from JSON file
with open('camera_params.json', 'r') as f:
    params = json.load(f)

# Extract intrinsics and extrinsics
color_intr = params["color_intrinsics"]
depth_intr = params["depth_intrinsics"]
extrinsics = params["extrinsics"]

R = np.array(extrinsics["rotation"]).reshape(3, 3)
T = np.array(extrinsics["translation"]).reshape(3, 1)

depth_camera_matrix = np.array([
    [depth_intr["fx"], 0, depth_intr["ppx"]],
    [0, depth_intr["fy"], depth_intr["ppy"]],
    [0, 0, 1]
])

color_camera_matrix = np.array([
    [color_intr["fx"], 0, color_intr["ppx"]],
    [0, color_intr["fy"], color_intr["ppy"]],
    [0, 0, 1]
])

# Load images
color_image = cv2.imread("color.png")
depth_image = cv2.imread("depth.png", cv2.IMREAD_UNCHANGED).astype(np.float32)

depth_image_meters = depth_image / 1000.0  # Convert mm to meters

height, width = depth_image.shape
u, v = np.meshgrid(np.arange(width), np.arange(height))
u = u.flatten()
v = v.flatten()
z = depth_image_meters.flatten()

valid = z > 0
u = u[valid]
v = v[valid]
z = z[valid]

x = (u - depth_intr["ppx"]) * z / depth_intr["fx"]
y = (v - depth_intr["ppy"]) * z / depth_intr["fy"]
depth_points = np.vstack((x, y, z))

color_points = R @ depth_points + T
uv_homog = color_camera_matrix @ color_points
u_color = (uv_homog[0] / uv_homog[2]).astype(int)
v_color = (uv_homog[1] / uv_homog[2]).astype(int)

valid = (u_color >= 0) & (u_color < color_intr["width"]) & (v_color >= 0) & (v_color < color_intr["height"])
u_color = u_color[valid]
v_color = v_color[valid]
z = z[valid]

aligned_depth_image = np.zeros((color_intr["height"], color_intr["width"]), dtype=np.float32)
aligned_depth_image[v_color, u_color] = z

original_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=255.0 / np.max(depth_image)), cv2.COLORMAP_JET)
aligned_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(aligned_depth_image, alpha=255.0 / np.max(aligned_depth_image)), cv2.COLORMAP_JET)

while True:
    cv2.imshow("Original RGB Image", color_image)
    cv2.imshow("Original Depth Image", original_depth_colormap)
    cv2.imshow("Aligned Depth Image", aligned_depth_colormap)
    combined = np.hstack((color_image, aligned_depth_colormap))
    cv2.imshow("Side-by-Side Comparison: RGB + Aligned Depth", combined)

    valid_points = color_points[:, valid].T
    valid_colors = color_image[v_color, u_color][:, ::-1] / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    pcd.colors = o3d.utility.Vector3dVector(valid_colors)

    o3d.visualization.draw_geometries([pcd], window_name="3D Point Cloud", width=800, height=600)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()