#!/usr/bin/env python3

# World coordinate lane detection with memorized curves - Simple approach

import rospy
import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
import tf.transformations as tf
import matplotlib.pyplot as plt
import traceback
from threading import Lock
from scipy.spatial.distance import cdist
from collections import deque

# Global variables
bridge = CvBridge()
data_lock = Lock()

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
        self.last_updated = rospy.Time.now()
        
        if self.valid:
            self.calculate_properties()

    def calculate_properties(self):
        if len(self.points) == 0:
            return
            
        x_vals = self.points[:, 0]
        y_vals = self.points[:, 1]
        
        self.x_avg = np.mean(x_vals)
        self.y_avg = np.mean(y_vals)
        
        try:
            if len(x_vals) >= 3:
                self.curve_coeffs = np.polyfit(x_vals, y_vals, 2)
                y_pred = np.polyval(self.curve_coeffs, x_vals)
                mse = np.mean((y_vals - y_pred) ** 2)
                self.confidence = 1.0 / (1.0 + mse)
            else:
                self.curve_coeffs = np.polyfit(x_vals, y_vals, 1)
                self.confidence = 0.5
        except (np.RankWarning, np.linalg.LinAlgError):
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
        if len(self.curves) < 2:
            return []
        
        sorted_curves = sorted(self.curves, key=lambda c: c.label_noY)
        left_curve = sorted_curves[0]
        right_curve = sorted_curves[-1]

        if len(left_curve.world_points) == 0 or len(right_curve.world_points) == 0:
            return []

        left_pts = np.array(left_curve.world_points)
        right_pts = np.array(right_curve.world_points)

        num_samples = min(len(left_pts), len(right_pts), 20)
        left_indices = np.linspace(0, len(left_pts)-1, num_samples).astype(int)
        right_indices = np.linspace(0, len(right_pts)-1, num_samples).astype(int)

        center_line = []
        for l_idx, r_idx in zip(left_indices, right_indices):
            lp = left_pts[l_idx]
            rp = right_pts[r_idx]
            center_pt = [(lp[0] + rp[0]) / 2.0, (lp[1] + rp[1]) / 2.0, (lp[2] + rp[2]) / 2.0]
            center_line.append(center_pt)

        return center_line
    
    def get_sample_center_points(self, center_line, sample_distance=0.5):
        if not center_line or len(center_line) < 3:
            return None 

        x_vals = [pt[0] for pt in center_line]
        y_vals = [pt[1] for pt in center_line]

        try:
            coeffs = np.polyfit(x_vals, y_vals, min(2, len(x_vals)-1))
            
            x_max = max(x_vals)
            x_min = min(x_vals)
            x_samples = np.arange(x_min, x_max, step=sample_distance)
            
            sampled_points = []
            for x in x_samples:
                y = np.polyval(coeffs, x)
                sampled_points.append([x, y, 0.0])

            return sampled_points
        except np.linalg.LinAlgError:
            rospy.logwarn("Failed to fit polynomial to center line")
            return None

class CurveTracker:
    def __init__(self, max_frames=5, match_threshold=2.0):
        self.frames = deque(maxlen=max_frames)
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

        for new_curve in new_frame.curves:
            match = self.match_curve(new_curve, past_curves, used_curves)
            if match:
                new_curve.curve_id = match.curve_id
                used_curves.add(match)
            else:
                new_curve.curve_id = self.new_curve_id
                self.new_curve_id += 1

        self.frames.append(new_frame)

class MemorizedCurvesWorldFrame:
    def __init__(self, max_distance=2.0, decay_time=30.0):
        self.memory = {}  # curve_id -> list of world points
        self.curve_timestamps = {}  # curve_id -> last update time
        self.max_distance = max_distance
        self.decay_time = decay_time

    def update_memory(self, robot_pose, frame):
        if robot_pose is None:
            rospy.logwarn("No robot pose available for memory update")
            return
            
        # Transform current frame curves to world coordinates
        self.transform_to_world(robot_pose, frame)
        current_time = rospy.Time.now()
        
        # Update memory with new curves
        for curve in frame.curves:
            if curve.curve_id is None or not curve.world_points:
                continue

            if curve.curve_id in self.memory:
                # Merge with existing curve
                old_points = self.memory[curve.curve_id]
                merged_points = self.merge_points_weighted(old_points, curve.world_points)
                self.memory[curve.curve_id] = merged_points
            else:
                # New curve
                self.memory[curve.curve_id] = list(curve.world_points)
            
            self.curve_timestamps[curve.curve_id] = current_time

        # Remove old curves
        self.cleanup_old_curves(current_time)

    def transform_to_world(self, robot_pose, frame):
        try:
            position = robot_pose.pose.position
            orientation = robot_pose.pose.orientation

            # Extract yaw (rotation around Z-axis) from quaternion
            q = [orientation.x, orientation.y, orientation.z, orientation.w]
            roll, pitch, yaw = tf.euler_from_quaternion(q)

            # Robot position in world
            x_robot = position.x
            y_robot = position.y * -1  # Convert to our coordinate system

            # Rotation matrix (2D)
            R = np.array([
                [np.cos(-yaw), -np.sin(-yaw)],
                [np.sin(-yaw),  np.cos(-yaw)]
            ])

            for curve in frame.curves:
                curve.world_points = []
                for pt in curve.points:
                    # pt is [x, y, z] in robot frame
                    pt_robot = np.array([pt[0], pt[1]])  # x, y in robot frame

                    # Apply 2D rotation and translation to get world coordinates
                    pt_world = R @ pt_robot + np.array([x_robot, y_robot])

                    # Save as world coordinate [x_world, y_world, z_robot]
                    curve.world_points.append([pt_world[0], pt_world[1], pt[2]])
                    
        except Exception as e:
            rospy.logerr(f"Transform error: {e}")

    def merge_points_weighted(self, old_points, new_points, 
                            new_weight=0.7, old_weight=0.3, 
                            distance_threshold=0.5):
        """
        Merge old and new curve points with weighted averaging
        """
        if not old_points:
            return list(new_points)
        if not new_points:
            return list(old_points)

        old_array = np.array([[pt[0], pt[1]] for pt in old_points])
        new_array = np.array([[pt[0], pt[1]] for pt in new_points])
        
        # Calculate distance matrix
        distances = cdist(new_array, old_array)
        
        merged_points = []
        used_old_indices = set()
        
        # Process new points
        for i, new_pt in enumerate(new_points):
            min_dist_idx = np.argmin(distances[i])
            min_dist = distances[i][min_dist_idx]
            
            if min_dist < distance_threshold and min_dist_idx not in used_old_indices:
                # Merge with closest old point (weighted average)
                old_pt = old_points[min_dist_idx]
                merged_pt = [
                    new_weight * new_pt[0] + old_weight * old_pt[0],
                    new_weight * new_pt[1] + old_weight * old_pt[1],
                    new_weight * new_pt[2] + old_weight * old_pt[2]
                ]
                merged_points.append(merged_pt)
                used_old_indices.add(min_dist_idx)
            else:
                # Add new point as is
                merged_points.append(list(new_pt))
        
        # Add unused old points
        for i, old_pt in enumerate(old_points):
            if i not in used_old_indices:
                merged_points.append(list(old_pt))
        
        return merged_points

    def cleanup_old_curves(self, current_time):
        """Remove curves that haven't been updated recently"""
        expired_curves = []
        for curve_id, timestamp in self.curve_timestamps.items():
            if (current_time - timestamp).to_sec() > self.decay_time:
                expired_curves.append(curve_id)
        
        for curve_id in expired_curves:
            if curve_id in self.memory:
                del self.memory[curve_id]
                rospy.loginfo(f"Removed expired curve ID: {curve_id}")
            if curve_id in self.curve_timestamps:
                del self.curve_timestamps[curve_id]

    def get_curves_near_robot(self, robot_pose, max_distance=15.0):
        """Get memorized curves that are near the current robot position"""
        if robot_pose is None:
            return {}
            
        robot_x = robot_pose.pose.position.x
        robot_y = robot_pose.pose.position.y * -1
        
        nearby_curves = {}
        
        for curve_id, points in self.memory.items():
            if not points:
                continue
                
            # Check if any point in the curve is within max_distance
            curve_points = np.array(points)
            distances = np.sqrt((curve_points[:, 0] - robot_x)**2 + (curve_points[:, 1] - robot_y)**2)
            
            if np.any(distances <= max_distance):
                # Filter points to only include those within distance
                nearby_indices = distances <= max_distance
                nearby_points = curve_points[nearby_indices].tolist()
                nearby_curves[curve_id] = nearby_points
                
        return nearby_curves

# ROS callback functions
def camera_info_callback(msg):
    with data_lock:
        camera_data.rgb_frame_size = (msg.height, msg.width)
        camera_data.camera_params.update({
            'fx': msg.K[0], 'fy': msg.K[4],
            'cx': msg.K[2], 'cy': msg.K[5]
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

# Processing functions
def separate_line_masks(binary_mask, min_area=300):
    if binary_mask is None or binary_mask.size == 0:
        return []
        
    binary_mask = binary_mask.astype(np.uint8)
    
    try:
        gpu_mask = cv2.cuda_GpuMat()
        gpu_mask.upload(binary_mask)
        _, gpu_mask = cv2.cuda.threshold(gpu_mask, 1, 255, cv2.THRESH_BINARY)
        binary_mask = gpu_mask.download()
    except:
        _, binary_mask = cv2.threshold(binary_mask, 1, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3,3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    separated_lines = []
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            line_image = np.zeros_like(binary_mask)
            cv2.drawContours(line_image, [contour], -1, 255, thickness=cv2.FILLED)
            separated_lines.append(line_image)
    
    return separated_lines

def extract_line_points(separated_lines, max_points_per_line=100):
    point_lists = []
    
    for line_mask in separated_lines:
        try:
            skeleton = cv2.ximgproc.thinning(line_mask)
            points = cv2.findNonZero(skeleton)
            
            if points is not None:
                points_2d = points.reshape(-1, 2)
                
                if len(points_2d) > max_points_per_line:
                    indices = np.linspace(0, len(points_2d)-1, max_points_per_line).astype(int)
                    points_2d = points_2d[indices]
                
                point_lists.append(points_2d)
        except Exception as e:
            rospy.logwarn(f"Error processing line mask: {e}")
            continue
    
    return point_lists

def convert_2d_to_3d(point_lists, depth_image, max_distance=12.0):
    if depth_image is None:
        return []
        
    params = camera_data.camera_params
    fx, fy, cx, cy = params['fx'], params['fy'], params['cx'], params['cy']
    
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
            
            if (0 <= x_img < depth_resized.shape[1] and 0 <= y_img < depth_resized.shape[0]):
                depth = depth_resized[y_img, x_img]
                
                if depth > 0 and not np.isnan(depth) and depth <= max_distance:
                    x_cam = depth 
                    y_cam = (x_img - cx) * depth / fx 
                    z_cam = (y_img - cy) * depth / fy

                    robot_point = np.array([x_cam, y_cam, z_cam])
                    
                    if not (np.isnan(robot_point).any() or np.isinf(robot_point).any()):
                        points_3d.append(robot_point)
        
        if len(points_3d) > 5:
            lines_3d.append(points_3d)
    
    return lines_3d

def publish_reference_path(sample_center_line, pub):
    """Publish reference path for path following"""
    if sample_center_line and len(sample_center_line) > 0:
        msg = Float32MultiArray()
        flattened = []
        for pt in sample_center_line:
            flattened.extend([pt[0], pt[1], pt[2]])
        msg.data = flattened
        pub.publish(msg)

def main():
    rospy.init_node('world_frame_memorized_lane_node')
    rospy.loginfo("World frame memorized lane node started")

    # Subscribers
    sub_mask = rospy.Subscriber('/wheeler/lane_node/line_mask', Image, callback=mask_callback)
    sub_depth = rospy.Subscriber('/zed2i/zed_node/depth/depth_registered', Image, callback=depth_callback)
    sub_pose = rospy.Subscriber('/zed2i/zed_node/pose', PoseStamped, callback=pose_callback)
    
    # Publishers
    refPath_pub = rospy.Publisher('/wheeler/ref_path/points', Float32MultiArray, queue_size=10)

    # Initialize components
    curve_tracker = CurveTracker(max_frames=5, match_threshold=2.0)
    memorized_curves = MemorizedCurvesWorldFrame(max_distance=2.0, decay_time=30.0)

    # Visualization setup
    plt.ion()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Store robot path for visualization
    robot_path_x = []
    robot_path_y = []
    
    rate = rospy.Rate(30)
    frame_count = 0
    
    rospy.loginfo("Starting world frame memorized processing loop")
    
    while not rospy.is_shutdown():
        start_time = time.time()
        
        try:
            with data_lock:
                line_mask = camera_data.line_mask.copy() if camera_data.line_mask is not None else None
                depth_data = camera_data.depth_data.copy() if camera_data.depth_data is not None else None
                robot_pose = camera_data.robot_pose
            
            if line_mask is None or depth_data is None or robot_pose is None:
                rospy.logwarn_throttle(1.0, "Waiting for camera data...")
                rate.sleep()
                continue
            
            frame_count += 1
            if frame_count <= 100:
                continue

            # Store robot path for visualization
            robot_x = robot_pose.pose.position.x
            robot_y = robot_pose.pose.position.y * -1
            robot_path_x.append(robot_x)
            robot_path_y.append(robot_y)
            
            # Keep only recent path points (last 1000 points)
            if len(robot_path_x) > 1000:
                robot_path_x = robot_path_x[-1000:]
                robot_path_y = robot_path_y[-1000:]

            # Process pipeline
            separated_lines = separate_line_masks(line_mask, min_area=300)
            point_lists = extract_line_points(separated_lines, max_points_per_line=50)
            lines_3d = convert_2d_to_3d(point_lists, depth_data)
            
            # Create frame with detected curves
            current_frame = Frame()
            for line_points in lines_3d:
                curve = Curve(line_points)
                current_frame.add_curve(curve)
            
            # Track curves and assign IDs
            current_frame.classify_curves()
            curve_tracker.add_frame(current_frame)
            
            # Update memorized curves
            memorized_curves.update_memory(robot_pose, current_frame)
            
            # Generate center line from current frame
            center_line = current_frame.get_center_line()
            sample_center_line = current_frame.get_sample_center_points(center_line)
            
            # Publish reference path
            publish_reference_path(sample_center_line, refPath_pub)

            # Get nearby memorized curves for visualization
            nearby_memorized = memorized_curves.get_curves_near_robot(robot_pose, max_distance=20.0)

            # Visualization
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink']
            
            # Plot 1: Current frame curves (world coordinates)
            ax1.clear()
            for curve in current_frame.curves:
                if len(curve.world_points) > 0:
                    points_arr = np.array(curve.world_points)
                    x = points_arr[:, 0]  # world x
                    y = points_arr[:, 1]  # world y
                    color = colors[curve.curve_id % len(colors)] if curve.curve_id is not None else 'gray'
                    ax1.scatter(x, y, c=color, s=15, alpha=0.8, label=f'Current ID:{curve.curve_id}')

            # Plot current center line
            if sample_center_line and len(sample_center_line) > 0:
                center_x = [pt[0] for pt in sample_center_line]
                center_y = [pt[1] for pt in sample_center_line]
                ax1.plot(center_x, center_y, 'r-', linewidth=3, label='Center Line')

            # Plot robot position
            ax1.scatter(robot_x, robot_y, c='black', s=100, marker='s', label='Robot')
            
            ax1.set_xlabel('World X (Forward)')
            ax1.set_ylabel('World Y (Left)')
            ax1.set_title('Current Frame - World Coordinates')
            ax1.grid(True)
            ax1.legend()
            ax1.axis('equal')

            # Plot 2: All memorized curves (world coordinates)
            ax2.clear()
            for curve_id, points in memorized_curves.memory.items():
                if len(points) > 0:
                    points_arr = np.array(points)
                    x = points_arr[:, 0]  # world x
                    y = points_arr[:, 1]  # world y
                    color = colors[curve_id % len(colors)]
                    ax2.scatter(x, y, c=color, s=5, alpha=0.6, label=f'Memory ID:{curve_id}')

            # Plot robot path
            if len(robot_path_x) > 1:
                ax2.plot(robot_path_x, robot_path_y, 'k--', linewidth=2, alpha=0.7, label='Robot Path')
            
            # Plot robot position
            ax2.scatter(robot_x, robot_y, c='black', s=100, marker='s', label='Robot')

            ax2.set_xlabel('World X (Forward)')
            ax2.set_ylabel('World Y (Left)')
            ax2.set_title('All Memorized Curves - World Coordinates')
            ax2.grid(True)
            ax2.legend()
            ax2.axis('equal')

            # Plot 3: Nearby memorized curves (zoomed view around robot)
            ax3.clear()
            for curve_id, points in nearby_memorized.items():
                if len(points) > 0:
                    points_arr = np.array(points)
                    x = points_arr[:, 0]  # world x
                    y = points_arr[:, 1]  # world y
                    color = colors[curve_id % len(colors)]
                    ax3.scatter(x, y, c=color, s=10, alpha=0.7, label=f'Near ID:{curve_id}')

            # Plot recent robot path
            if len(robot_path_x) > 50:
                recent_x = robot_path_x[-50:]
                recent_y = robot_path_y[-50:]
                ax3.plot(recent_x, recent_y, 'k--', linewidth=2, alpha=0.8, label='Recent Path')
            
            # Plot robot position and orientation
            ax3.scatter(robot_x, robot_y, c='red', s=150, marker='s', label='Robot')
            
            # Draw robot orientation
            q = [robot_pose.pose.orientation.x, robot_pose.pose.orientation.y, 
                 robot_pose.pose.orientation.z, robot_pose.pose.orientation.w]
            roll, pitch, yaw = tf.euler_from_quaternion(q)
            arrow_length = 2.0
            arrow_x = robot_x + arrow_length * np.cos(-yaw)  # Note: -yaw for our coordinate system
            arrow_y = robot_y + arrow_length * np.sin(-yaw)
            ax3.arrow(robot_x, robot_y, arrow_x - robot_x, arrow_y - robot_y,
                     head_width=0.5, head_length=0.3, fc='red', ec='red', alpha=0.8)

            ax3.set_xlabel('World X (Forward)')
            ax3.set_ylabel('World Y (Left)')
            ax3.set_title('Nearby Memorized Curves (Robot View)')
            ax3.set_xlim([robot_x - 10, robot_x + 15])
            ax3.set_ylim([robot_y - 8, robot_y + 8])
            ax3.grid(True)
            ax3.legend()
            ax3.axis('equal')

            # Plot 4: Statistics and info
            ax4.clear()
            
            # Show curve statistics
            total_curves = len(memorized_curves.memory)
            nearby_curves = len(nearby_memorized)
            current_curves = len(current_frame.curves)
            
            stats_text = [
                f"Total memorized curves: {total_curves}",
                f"Nearby curves: {nearby_curves}", 
                f"Currently detected: {current_curves}",
                f"Robot position: ({robot_x:.1f}, {robot_y:.1f})",
                f"Frame: {frame_count}"
            ]
            
            for i, text in enumerate(stats_text):
                ax4.text(0.05, 0.9 - i*0.15, text, transform=ax4.transAxes, fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            
            # Show active curve IDs
            active_ids = list(memorized_curves.memory.keys())
            current_ids = [c.curve_id for c in current_frame.curves if c.curve_id is not None]
            
            ax4.text(0.05, 0.3, f"Active IDs: {active_ids}", transform=ax4.transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            ax4.text(0.05, 0.15, f"Current IDs: {current_ids}", transform=ax4.transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
            
            ax4.set_title('Statistics and Information')
            ax4.axis('off')

            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)

            # OpenCV displays
            cv2.imshow('Line Mask', line_mask)
            if np.max(depth_data) > 0:
                cv2.imshow('Depth', depth_data / np.max(depth_data))
            
            for i, line_img in enumerate(separated_lines[:3]):
                cv2.imshow(f"Separated Line {i}", line_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Performance monitoring
            processing_time = time.time() - start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0
            
            if frame_count % 30 == 0:  # Print every second
                rospy.loginfo(f"Processing FPS: {fps:.1f}, Total memorized curves: {len(memorized_curves.memory)}, Current curves: {len(current_frame.curves)}")
                
                # Print curve IDs for debugging
                active_ids = list(memorized_curves.memory.keys())
                current_ids = [c.curve_id for c in current_frame.curves if c.curve_id is not None]
                rospy.loginfo(f"Active memorized IDs: {active_ids}")
                rospy.loginfo(f"Current detected IDs: {current_ids}")
                
        except Exception as e:
            rospy.logerr(f"Processing error: {e}")
            traceback.print_exc()
        
        rate.sleep()
    
    plt.ioff()
    cv2.destroyAllWindows()
    rospy.loginfo("World frame memorized lane node shutting down")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass