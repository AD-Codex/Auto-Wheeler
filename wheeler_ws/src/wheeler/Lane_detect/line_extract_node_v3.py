#!/usr/bin/env python3

# plot memorize curve with curve in base frame
# The Method:
# 1. Find Nearest Points

# Compare curve 1 (old) points with curve 2 (new) points
# Find the closest pairs within a distance threshold

# 2. Weighted Averaging

# For matched points: averaged_point = new_weight * new_point + old_weight * old_point
# Default: new_weight = 0.7, old_weight = 0.3 (gives more weight to new data)

# 3. Handle All Points

# Matched points: Get weighted average
# Unmatched old points: Keep them (preserves old curve data)
# Unmatched new points: Add them (captures new curve extensions)


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
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tf.transformations as tf
import gc
import matplotlib.pyplot as plt
import traceback
from threading import Lock
import numpy as np
from scipy.spatial.distance import cdist


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

    def get_center_line(self):
        if not self.curves:
            return
        
        # Sort by label_noY to get leftmost and rightmost
        sorted_curves = sorted(self.curves, key=lambda c: c.label_noY)

        left_curve = sorted_curves[0]
        right_curve = sorted_curves[-1]

        # Ensure both have enough points
        if len(left_curve.points) == 0 or len(right_curve.points) == 0:
            return []

        # Interpolate to align the points (if needed)
        left_pts = np.array(left_curve.points)
        right_pts = np.array(right_curve.points)

        # Resample both to the same number of points
        num_samples = min(len(left_pts), len(right_pts))
        left_pts_resampled = np.linspace(0, len(left_pts)-1, num_samples).astype(int)
        right_pts_resampled = np.linspace(0, len(right_pts)-1, num_samples).astype(int)

        center_line = []
        for l_idx, r_idx in zip(left_pts_resampled, right_pts_resampled):
            lp = left_pts[l_idx]
            rp = right_pts[r_idx]
            center_pt = [(lp[0] + rp[0]) / 2.0, (lp[1] + rp[1]) / 2.0, (lp[2] + rp[2]) / 2.0]
            center_line.append(center_pt)

        return center_line
    
    def get_sample_center_points(self, center_line):
        if not center_line or len(center_line) < 3:
            return None 

        x_vals = [pt[0] for pt in center_line]
        y_vals = [pt[1] for pt in center_line]

        coeffs = np.polyfit(x_vals, y_vals, 2)

        sampled_points = []
        x_vals = np.arange(0.0, max(x_vals), step=0.5)
        for x in x_vals:
            y = np.polyval(coeffs, x)
            sampled_points.append([x, y])  # z = 0.0

        return sampled_points

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

class MemorizedCurves:
    def __init__(self, max_distance=2.0):
        self.memory = {}
        self.max_distance = max_distance

    def update_memory(self, robot_pose, frame):
        # Update memory with new curves from the current frame
        if robot_pose is None:
            rospy.logwarn("No robot pose available for memory update")
            return
            
        self.transform_to_world(robot_pose, frame)
        
        for curve in frame.curves:
            if curve.curve_id is None or not curve.world_points:
                continue

            if curve.curve_id in self.memory:
                # Merge with existing curve
                old_points = self.memory[curve.curve_id]
                merged_points = self.merge_points(old_points, curve.world_points)
                self.memory[curve.curve_id] = merged_points
            else:
                # New curve
                self.memory[curve.curve_id] = list(curve.world_points)

    def transform_to_world(self, robot_pose, frame):
        try:
            # position = robot_pose.pose.position
            # orientation = robot_pose.pose.orientation

            # q = [orientation.x, orientation.y, orientation.z, orientation.w]
            # T = tf.quaternion_matrix(q)
            # T[0:3, 3] = [position.x, -position.y, position.z]

            # for curve in frame.curves:
            #     curve.world_points = []
            #     for pt in curve.points:
            #         # Transform from camera to robot frame
            #         pt_robot = np.array([pt[0], pt[1], pt[2], 1.0])
            #         pt_world = T @ pt_robot
            #         # Convert back to robot frame convention
            #         world_coord = [pt_world[0], pt_world[1], pt_world[2]]
            #         curve.world_points.append(world_coord)

            position = robot_pose.pose.position
            orientation = robot_pose.pose.orientation

            # Extract yaw (rotation around Z-axis) from quaternion
            q = [orientation.x, orientation.y, orientation.z, orientation.w]
            roll, pitch, yaw = tf.euler_from_quaternion(q)
            # print(ptich, roll, yaw)

            # Camera (robot) position in world
            x_cam = position.x
            y_cam = position.y *-1

            # Rotation matrix (2D)
            R = np.array([
                [np.cos(-yaw), -np.sin(-yaw)],
                [np.sin(-yaw),  np.cos(-yaw)]
            ])

            for curve in frame.curves:
                curve.world_points = []
                for pt in curve.points:
                    # pt assumed to be [x, y] in camera frame
                    pt_cam = np.array([pt[0], pt[1]])  # or reverse if your pt format is different

                    # Apply 2D rotation and translation
                    # pt_world = R @ pt_cam
                    pt_world = R @ pt_cam + np.array([x_cam, y_cam])

                    # Save as 2D world coordinate
                    curve.world_points.append([pt_world[0], pt_world[1], 0.0])
                    
        except Exception as e:
            rospy.logerr(f"Transform error: {e}")

    def merge_points(self, old_points, new_points, min_threshold=0.2, max_threshold=1.0):
        if not old_points:
            return list(new_points)
        if not new_points:
            return list(old_points)

        new_array = np.array([[pt[0], pt[1]] for pt in new_points])
        merged = list(new_points)  # Start with new_points as the base

        for old_pt in old_points:
            old_pt_arr = np.array([old_pt[0], old_pt[1]])
            distances = np.linalg.norm(new_array - old_pt_arr, axis=1)

            # If old point is not close to any new point, add it
            if np.min(distances) > min_threshold:
                merged.append(old_pt)

        return merged





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
                    x_cam = depth 
                    y_cam = (x_img - cx) * depth / fx 
                    z_cam = (y_img - cy) * depth / fy

                    # distance limit to 12 meters
                    if ( x_cam <= 12) :
                        robot_point = np.array([x_cam, y_cam, z_cam])
                        
                        if not (np.isnan(robot_point).any() or np.isinf(robot_point).any()):
                            points_3d.append(robot_point)
        
        if len(points_3d) > 0:
            lines_3d.append(points_3d)
    
    return lines_3d





def main():
    rospy.init_node('line_extract_node')

    rospy.loginfo("Line extract node started")
    rospy.loginfo("Published topics:")
    

    sub_mask = rospy.Subscriber('/wheeler/lane_node/line_mask', Image, callback = mask_callback)
    sub_depth = rospy.Subscriber('/zed2i/zed_node/depth/depth_registered', Image, callback = depth_callback)
    sub_pose = rospy.Subscriber('/zed2i/zed_node/pose', PoseStamped, callback = pose_callback)
    # sub_camera_info = rospy.Subscriber("/zed2i/zed_node/rgb/camera_info", CameraInfo, callback = camera_info_callback)
    
    refPath_pub = rospy.Publisher('/wheeler/ref_path/points', Float32MultiArray, queue_size=10)

    curve_tracker = CurveTracker(max_frames=5, match_threshold=2000.0)
    memorized_curves = MemorizedCurves(max_distance=50)


    # Initialize visualization
    plt.ion()
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')

    fig, ax = plt.subplots()  # 2D plot instead of 3D
    fig2, ax2 = plt.subplots()  # 2D plot instead of 3D
    
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

            # neglect first 10s frames
            if frame_count <= 300:
                continue
            

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
            memorized_curves.update_memory(robot_pose, current_frame)
            center_line = current_frame.get_center_line()
            sample_center_line = current_frame.get_sample_center_points(center_line)

            print(sample_center_line)


            # # 2D plot of curve point
            ax.clear()
            for curve in current_frame.curves:
                x = [point[1] for point in curve.points]
                y = [point[0] for point in curve.points]
                ax.scatter(x, y)
                ax.text(x[-1], y[-1], f"id {curve.curve_id, curve.label_noY}")

            # Plot center line
            
            if sample_center_line and len(sample_center_line) > 0:
                center_x = [pt[1] for pt in sample_center_line]  # lateral
                center_y = [pt[0] for pt in sample_center_line]  # forward
                ax.plot(center_x, center_y, 'r-', linewidth=2, label='Center Line')

            ax.set_xlabel('X (Right)')
            ax.set_ylabel('Y (Forward)')
            ax.set_ylim([0,20])
            ax.set_xlim([-10,10])
            plt.grid(True)
            plt.draw()


            # # 2D plot of curve world point
            # ax2.clear()
            # for curve in current_frame.curves:
            #     x = [point[1] for point in curve.world_points]
            #     y = [point[0] for point in curve.world_points]
            #     ax2.scatter(x, y)
            #     ax2.text(x[-1], y[-1], f"id {curve.curve_id}")

            # ax2.set_xlabel('X (Forward)')
            # ax2.set_ylabel('Y (Left)')
            # ax2.set_ylim([0,20])
            # ax2.set_xlim([-10,10])
            # plt.grid(True)
            # plt.draw()



            ax2.clear()
            for curve_id, points in memorized_curves.memory.items():
                points = np.array(points)
                x = [pt[1] for pt in points]
                y = [pt[0] for pt in points]
                ax2.scatter(x, y)
                ax2.text(x[-1], y[-1], f"id {curve_id}")

            ax2.set_xlabel('X (Forward)')
            ax2.set_ylabel('Y (Left)')
            ax2.set_ylim([0,20])
            ax2.set_xlim([-10,10])
            plt.grid(True)
            plt.draw()


            # # plot x,y,z 3D ----------------------------------
            # ax.clear()
            # for curve in current_frame.curves:
            #     # print("curve ids", curve.curve_id)
            #     x = [point[0] for point in curve.points]
            #     y = [point[1] for point in curve.points]
            #     z = [point[2] for point in curve.points]

            #     ax.scatter(x,y,z)
            #     ax.text(x[-1], y[-1], 0, f"id {curve.curve_id}")
            # ax.set_xlabel('X (Forward)')
            # ax.set_ylabel('Y (left)')
            # ax.set_zlabel('z (Height)')
            # plt.grid(True)
            # plt.draw()

            # # plot x,y,z 3D ----------------------------------
            # ax.clear()
            # for curve in current_frame.curves:
            #     # print("curve ids", curve.curve_id)
            #     x = [point[0] for point in curve.world_points]
            #     y = [point[1] for point in curve.world_points]
            #     z = [point[2] for point in curve.points]

            #     ax.scatter(x,y,z)
            #     ax.text(x[-1], y[-1], 0, f"id {curve.curve_id}")
            # ax.set_xlabel('X (Forward)')
            # ax.set_ylabel('Y (left)')
            # ax.set_zlabel('z (Height)')
            # plt.grid(True)
            # plt.draw()
            


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