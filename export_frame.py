import pyrealsense2 as rs
import numpy as np
import cv2
import json

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
profile = pipeline.start(config)

# Get intrinsics and extrinsics
color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
color_intr = color_profile.get_intrinsics()
depth_intr = depth_profile.get_intrinsics()
extrinsics = depth_profile.get_extrinsics_to(color_profile)

frame_count = 0

while True:
    frame_count = frame_count + 1

    # Capture a frame
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if frame_count < 200:
        continue

    # Convert frames to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Save images
    cv2.imwrite("color.png", color_image)
    cv2.imwrite("depth.png", depth_image)  # 16-bit PNG

    # Save intrinsics and extrinsics
    camera_params = {
        "color_intrinsics": {
            "width": color_intr.width,
            "height": color_intr.height,
            "fx": color_intr.fx,
            "fy": color_intr.fy,
            "ppx": color_intr.ppx,
            "ppy": color_intr.ppy,
            "coeffs": color_intr.coeffs
        },
        "depth_intrinsics": {
            "width": depth_intr.width,
            "height": depth_intr.height,
            "fx": depth_intr.fx,
            "fy": depth_intr.fy,
            "ppx": depth_intr.ppx,
            "ppy": depth_intr.ppy,
            "coeffs": depth_intr.coeffs
        },
        "extrinsics": {
            "rotation": extrinsics.rotation,
            "translation": extrinsics.translation
        }
    }

    with open("camera_params.json", "w") as f:
        json.dump(camera_params, f, indent=4)

    print("Saved color image, depth image, and camera parameters.")

    # Stop streaming
    pipeline.stop()

    break