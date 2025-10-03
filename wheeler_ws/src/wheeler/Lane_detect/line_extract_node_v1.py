#!/usr/bin/env python3

# plot curve and curve polynomial function in camera base

# ZED CAMERA COORDINATE SYSTEM (LEFT-HANDED)
# ------------------------------------------
# - X: Right
# - Y: Down
# - Z: Forward (depth direction)

# Robot Frame (Right-Handed):
# ----------------------------
# - X: Forward
# - Y: Left
# - Z: Up

import rospy
import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tf.transformations as tf
import gc
import matplotlib.pyplot as plt
import traceback
from threading import Lock


# Global variables
bridge = CvBridge()
data_lock = Lock()

# Data containers ------------------------------------------------------------
class CameraData:
    def __init__(self):
        self.line_mask = None
        self.depth_data = None
        self.robot_pose = None
        self.rgb_frame_size = (360, 640)
        self.camera_params = {
            'fx': 959.5588989257812 / 2,
            'fy': 959.5588989257812 / 2,
            'cx': 631.89208984375 / 2,
            'cy': 376.25347900390625 / 2
        }

camera_data = CameraData()



# lane curve classes ---------------------------------------------------------

class Curve:
    def __init__(self, points, min_points=5):
        self.points = np.array(points) if len(points) > 0 else np.array([])
        self.world_points = []
        self.curve_id = None
        self.curve_coeffs = []
        self.x_avg = 0
        self.y_avg = 0
        self.label_noX = 0
        self.label_noY = 0
        self.confidence = 0.0
        self.valid = len(points) >= min_points
        
        if self.valid:
            self.calculate_properties()

    def calculate_properties(self):
        if len(self.points) == 0:
            return
            
        x_vals = self.points[:, 0]
        y_vals = self.points[:, 1]
        
        # calcuate avg values
        self.x_avg = np.mean(x_vals)
        self.y_avg = np.mean(y_vals)
        
        # Fit polynomial
        try:
            if len(x_vals) >= 3:
                self.curve_coeffs = np.polyfit(x_vals, y_vals, 2)
                # Calculate confidence to profe curve fit
                y_pred = np.polyval(self.curve_coeffs, x_vals)
                mse = np.mean((y_vals - y_pred) ** 2)
                self.confidence = 1.0 / (1.0 + mse)
            else:
                self.curve_coeffs = np.polyfit(x_vals, y_vals, 1)
                self.confidence = 0.5
        except np.RankWarning:
            rospy.logwarn("Polynomial fitting failed for curve")
            self.valid = False

    def offset_to(self, other_curve):
        if not (self.valid and other_curve.valid):
            return float('inf')
        return np.sqrt((self.x_avg - other_curve.x_avg)**2 + (self.y_avg - other_curve.y_avg)**2)

class Frame:
    def __init__(self):
        self.curves = []
        self.timestamp = rospy.Time.now()

    def add_curve(self, curve):
        if curve.valid and len(curve.points) > 0:
            self.curves.append(curve)

    def classify_curves(self):
        if not self.curves:
            return
            
        # Sort by x position (forward direction)
        self.curves.sort(key=lambda c: c.x_avg)
        for idx, curve in enumerate(self.curves):
            curve.label_noX = idx

        # Sort by y position (lateral direction)
        self.curves.sort(key=lambda c: c.y_avg)
        for idx, curve in enumerate(self.curves):
            curve.label_noY = idx

class CurveTracker:
    def __init__(self, max_frames=5, match_threshold=50.0):
        self.frames = []
        self.max_frames = max_frames
        self.new_curve_id = 1
        self.match_threshold = match_threshold

    def get_past_curves(self):
        return [curve for frame in self.frames for curve in frame.curves]

    def match_curve(self, new_curve, past_curves, used_curves):
        best_match = None
        best_offset = float('inf')
        
        for past_curve in past_curves:
            if past_curve in used_curves:
                continue
                
            offset = new_curve.offset_to(past_curve)
            if offset < self.match_threshold and offset < best_offset:
                best_offset = offset
                best_match = past_curve
                
        return best_match

    def add_frame(self, new_frame):
        past_curves = self.get_past_curves()
        used_curves = set()

        # Match each new curve with past curves
        for new_curve in new_frame.curves:
            match = self.match_curve(new_curve, past_curves, used_curves)
            if match:
                new_curve.curve_id = match.curve_id
                used_curves.add(match)
            else:
                new_curve.curve_id = self.new_curve_id
                self.new_curve_id += 1

        # Add frame and maintain size limit
        self.frames.append(new_frame)
        if len(self.frames) > self.max_frames:
            self.frames.pop(0)


# ROS callback functions -----------------------------------------------------

def camera_info_callback(msg):
    with data_lock:
        camera_data.rgb_frame_size = (msg.height, msg.width)
        camera_data.camera_params.update({
            'fx': msg.K[0],
            'fy': msg.K[4],
            'cx': msg.K[2],
            'cy': msg.K[5]
        })

def mask_callback(msg):
    with data_lock:
        camera_data.line_mask = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")

def depth_callback(msg):
    with data_lock:
        camera_data.depth_data = bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")

def pose_callback(msg):
    with data_lock:
        camera_data.robot_pose = msg



# line process fn ------------------------------------------------------------

def separate_line_masks(binary_mask, min_area=500):
    # line seperate fn ( according to x axis)
    if binary_mask is None or binary_mask.size == 0:
        return []
        
    binary_mask = binary_mask.astype(np.uint8)
    
    # Use GPU acceleration if available
    try:
        gpu_mask = cv2.cuda_GpuMat()
        gpu_mask.upload(binary_mask)
        _, gpu_mask = cv2.cuda.threshold(gpu_mask, 1, 255, cv2.THRESH_BINARY)
        binary_mask = gpu_mask.download()
    except:
        _, binary_mask = cv2.threshold(binary_mask, 1, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    separated_lines = []
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            line_image = np.zeros_like(binary_mask)
            cv2.drawContours(line_image, [contour], -1, 255, thickness=cv2.FILLED)
            separated_lines.append(line_image)
    
    return separated_lines

def extract_line_points(separated_lines):
    # extract line point of line mask
    point_lists = []
    
    for line_mask in separated_lines:
        try:
            # Skeletonize the mask
            skeleton = cv2.ximgproc.thinning(line_mask)
            
            # Extract points
            points = cv2.findNonZero(skeleton)
            if points is not None:
                point_lists.append(points.reshape(-1, 2))
        except Exception as e:
            rospy.logwarn(f"Error processing line mask: {e}")
            continue
    
    return point_lists

def convert_2d_to_3d(point_lists, depth_image):
    # convert 2D coord to 3D coord
    if depth_image is None:
        return []
        
    params = camera_data.camera_params
    fx, fy, cx, cy = params['fx'], params['fy'], params['cx'], params['cy']
    
    # Resize depth to match mask dimensions
    try:
        if depth_image.shape != (360, 640):
            depth_resized = cv2.resize(depth_image, (640, 360), interpolation=cv2.INTER_LINEAR)
        else:
            depth_resized = depth_image
    except:
        return []
    
    lines_3d = []
    
    for points_2d in point_lists:
        points_3d = []
        
        for point in points_2d:
            x_img, y_img = point[0], point[1]
            
            # Check bounds
            if (0 <= x_img < depth_resized.shape[1] and 0 <= y_img < depth_resized.shape[0]):
                
                depth = depth_resized[y_img, x_img]
                
                if depth > 0 and not np.isnan(depth):
                    # Convert to 3D camera coordinates
                    x_cam = (x_img - cx) * depth / fx
                    y_cam = (y_img - cy) * depth / fy
                    z_cam = depth
                    
                    # Convert to robot frame (Z forward, Y left, X up becomes X forward, Y left, Z up)
                    robot_point = np.array([z_cam, -x_cam, -y_cam])
                    
                    if not (np.isnan(robot_point).any() or np.isinf(robot_point).any()):
                        points_3d.append(robot_point)
        
        if len(points_3d) > 0:
            lines_3d.append(points_3d)
    
    return lines_3d


# matplot visualizer ----------------------------------------------------------

def visualize_curves(frame, ax, memorized_curves=None, show_memorized=True, robot_pose=None):
    ax.clear()
    
    for curve in frame.curves:
        if not curve.valid or len(curve.points) == 0:
            continue
            
        # Plot curve points
        points = np.array(curve.points)
        ax.scatter(points[:, 0], points[:, 1], 0, alpha=0.6, s=10)
        
        # Plot fitted curve if available
        if len(curve.curve_coeffs) >= 2:
            try:
                x_min, x_max = points[:, 0].min(), points[:, 0].max()
                x_curve = np.linspace(x_min, x_max, 50)
                
                if len(curve.curve_coeffs) == 3:  # Quadratic
                    a, b, c = curve.curve_coeffs
                    y_curve = a * x_curve**2 + b * x_curve + c
                else:  # Linear
                    b, c = curve.curve_coeffs
                    y_curve = b * x_curve + c
                    
                ax.plot(x_curve, y_curve, 0, linewidth=2, alpha=0.8)
                
                # Add curve ID label
                if curve.curve_id is not None:
                    ax.text(x_curve[-1], y_curve[-1], 0, f"ID:{curve.curve_id}", 
                           fontsize=8)
            except Exception as e:
                rospy.logwarn(f"Visualization error: {e}")
    
    ax.set_xlabel('X (Forward)')
    ax.set_ylabel('Y (Left)')
    ax.set_zlabel('Z (Up)')
    ax.grid(True)
    plt.draw()
    plt.pause(0.001)  # Non-blocking update



def main():
    rospy.init_node('line_extract_node')

    rospy.loginfo("Line extract node started")
    rospy.loginfo("Published topics:")
    

    sub_mask = rospy.Subscriber("/robocop/lane_node/line_mask", Image, callback = mask_callback)
    sub_depth = rospy.Subscriber("/zed2i/zed_node/depth/depth_registered", Image, callback = depth_callback)
    sub_pose = rospy.Subscriber("/zed2i/zed_node/pose", PoseStamped, callback = pose_callback)
    # sub_camera_info = rospy.Subscriber("/zed2i/zed_node/left/camera_info", CameraInfo, callback = camera_info_callback)
    

    curve_tracker = CurveTracker(max_frames=5, match_threshold=2000.0)    

    # Initialize visualization
    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    rate = rospy.Rate(30)  # 30 Hz
    frame_count = 0
    

    rospy.loginfo("Starting main processing loop")
    
    while not rospy.is_shutdown():
        start_time = time.time()
        
        try:
            with data_lock:
                line_mask = camera_data.line_mask.copy() if camera_data.line_mask is not None else None
                depth_data = camera_data.depth_data.copy() if camera_data.depth_data is not None else None
                robot_pose = camera_data.robot_pose
            
            if line_mask is None or depth_data is None:
                rospy.logwarn_throttle(1.0, "Waiting for camera data...")
                rate.sleep()
                continue
            
            frame_count += 1
            

            # Process lane detection pipeline
            separated_lines = separate_line_masks(line_mask)
            point_lists = extract_line_points(separated_lines)
            lines_3d = convert_2d_to_3d(point_lists, depth_data)
            
            # Create frame with curves
            current_frame = Frame()
            for line_points in lines_3d:
                if len(line_points) > 0:
                    curve = Curve(line_points)
                    current_frame.add_curve(curve)
            
            # Process curves
            current_frame.classify_curves()
            curve_tracker.add_frame(current_frame)
            

            # Visualization
            if frame_count % 3 == 0:  # Update visualization every 3 frames
                visualize_curves(current_frame, ax)


            
            # Display images
            cv2.imshow('Line Mask', line_mask)
            cv2.imshow('Depth', depth_data / np.max(depth_data) if np.max(depth_data) > 0 else depth_data)
            
            # Display separated lines
            for i, line_img in enumerate(separated_lines[:5]):  # Limit to 5 windows
                cv2.imshow(f"Line {i}", line_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Performance monitoring
            processing_time = time.time() - start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0
            print(" Point extract FPS : ", fps)
            
                
        except Exception as e:
            rospy.logerr(f"Processing error: {e}")
            traceback.print_exc()
        
        rate.sleep()
    
    # Cleanup
    plt.ioff()
    cv2.destroyAllWindows()
    rospy.loginfo("Lane detection node shutting down")




if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass