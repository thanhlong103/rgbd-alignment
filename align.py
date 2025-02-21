import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
profile = pipeline.start(config)

# Get camera intrinsics and extrinsics
color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))

color_intr = color_profile.get_intrinsics()
depth_intr = depth_profile.get_intrinsics()
extrinsics = depth_profile.get_extrinsics_to(color_profile)

# Extrinsics: Rotation (R) and Translation (T)
R = np.array(extrinsics.rotation).reshape(3, 3)  # 3x3 rotation matrix
T = np.array(extrinsics.translation).reshape(3, 1)  # 3x1 translation vector

# Intrinsics: Camera matrices
depth_camera_matrix = np.array([
    [depth_intr.fx, 0, depth_intr.ppx],
    [0, depth_intr.fy, depth_intr.ppy],
    [0, 0, 1]
])

color_camera_matrix = np.array([
    [color_intr.fx, 0, color_intr.ppx],
    [0, color_intr.fy, color_intr.ppy],
    [0, 0, 1]
])

frame_count = 0 

try:
    while True:
        frame_count = frame_count + 1
        # Wait for frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        if frame_count < 100:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        color_image = np.asanyarray(color_frame.get_data())

        # Convert depth image to meters (RealSense depth is in mm)
        depth_image_meters = depth_image / 1000.0  # Convert mm to meters

        # Get pixel coordinates for the depth image
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
        x = (u - depth_intr.ppx) * z / depth_intr.fx
        y = (v - depth_intr.ppy) * z / depth_intr.fy
        depth_points = np.vstack((x, y, z))  # Shape: (3, N)

        # Transform points from depth to color camera coordinates
        color_points = R @ depth_points + T  # Shape: (3, N)

        # Project 3D points onto the color image plane
        uv_homog = color_camera_matrix @ color_points  # Shape: (3, N)
        u_color = (uv_homog[0] / uv_homog[2]).astype(int)
        v_color = (uv_homog[1] / uv_homog[2]).astype(int)

        # Filter points inside the color image bounds
        valid = (u_color >= 0) & (u_color < color_intr.width) & \
                (v_color >= 0) & (v_color < color_intr.height)

        u_color = u_color[valid]
        v_color = v_color[valid]
        z = z[valid]

        # 🌟 Create aligned depth image (warping depth to match color frame)
        aligned_depth_image = np.zeros((color_intr.height, color_intr.width), dtype=np.float32)
        aligned_depth_image[v_color, u_color] = z

        # Normalize and convert depth images to color maps
        original_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=255.0 / np.max(depth_image)), cv2.COLORMAP_JET)
        aligned_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(aligned_depth_image, alpha=255.0 / np.max(aligned_depth_image)), cv2.COLORMAP_JET)

        while True:
            combined_original = np.hstack((color_image, original_depth_colormap))
            cv2.imshow("Side-by-Side Comparison: RGB + Original Depth", original_depth_colormap)

            combined_aligned = np.hstack((color_image, aligned_depth_colormap))
            cv2.imshow("Side-by-Side Comparison: RGB + Aligned Depth", combined_aligned)

            # 🎯 Generate 3D Point Cloud
            valid_points = color_points[:, valid].T  # (N, 3)
            valid_colors = color_image[v_color, u_color][:, ::-1] / 255.0


            # Convert to Open3D format
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(valid_points)
            pcd.colors = o3d.utility.Vector3dVector(valid_colors)

            # Display the point cloud
            o3d.visualization.draw_geometries([pcd], window_name="3D Point Cloud", width=800, height=600)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
