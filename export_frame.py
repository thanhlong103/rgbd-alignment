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

frame_count = 0

try:
    while True:
        frame_count = frame_count + 1
        # Wait for frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError("Could not get frames from camera!")
        
        if frame_count < 200:
            continue

        # Convert frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())  # Shape: (480, 640, 3)
        depth_image = np.asanyarray(depth_frame.get_data())  # Shape: (480, 640), 16-bit depth

        # Save images
        cv2.imwrite("color.png", color_image)  # Save color frame
        cv2.imwrite("depth.png", depth_image)  # Save depth frame (16-bit)

        # Get intrinsics
        color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

        # Save intrinsics to JSON file
        intrinsics_data = {
            "color_intrinsics": {
                "width": color_intrinsics.width,
                "height": color_intrinsics.height,
                "fx": color_intrinsics.fx,
                "fy": color_intrinsics.fy,
                "ppx": color_intrinsics.ppx,
                "ppy": color_intrinsics.ppy,
                "coeffs": color_intrinsics.coeffs
            },
            "depth_intrinsics": {
                "width": depth_intrinsics.width,
                "height": depth_intrinsics.height,
                "fx": depth_intrinsics.fx,
                "fy": depth_intrinsics.fy,
                "ppx": depth_intrinsics.ppx,
                "ppy": depth_intrinsics.ppy,
                "coeffs": depth_intrinsics.coeffs
            }
        }

        with open("intrinsics.json", "w") as f:
            json.dump(intrinsics_data, f, indent=4)

        print("Color and depth frames saved!")
        print("Intrinsics saved to intrinsics.json")

        break

finally:
    pipeline.stop()
