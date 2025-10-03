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
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
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

        for i, x in enumerate(x_vals):
            y = np.polyval(coeffs, x)

            # Calculate theta (orientation angle)
            if i < len(x_vals) - 1:
                # Get next point
                x_next = x_vals[i + 1]
                y_next = np.polyval(coeffs, x_next)
                
                # Calculate angle from current point to next point
                dx = x_next - x
                dy = y_next - y
                theta = np.arctan2(dy, dx)
            else:
                # For the last point, use the same theta as the previous point
                if len(sampled_points) > 0:
                    theta = sampled_points[-1][2]
                else:
                    theta = 0.0
            
            sampled_points.append([x, y, theta])

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
        self.center_line_memory = []  # Store previous center line in world frame
        self.lane_width = 4

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
            position = robot_pose.pose.position
            orientation = robot_pose.pose.orientation

            # Extract yaw from quaternion
            q = [orientation.x, orientation.y, orientation.z, orientation.w]
            roll, pitch, yaw = tf.euler_from_quaternion(q)

            # Robot position in world
            x_robot = position.x
            y_robot = position.y * -1

            # 2D Rotation matrix
            R = np.array([
                [np.cos(-yaw), -np.sin(-yaw)],
                [np.sin(-yaw),  np.cos(-yaw)]
            ])

            for curve in frame.curves:
                curve.world_points = []
                for pt in curve.points:
                    pt_robot = np.array([pt[0], pt[1]])
                    pt_world = R @ pt_robot + np.array([x_robot, y_robot])
                    curve.world_points.append([pt_world[0], pt_world[1], pt[2]])
                    
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

    def transform_world_to_robot(self, robot_pose, world_points):
        """Transform points from world frame back to robot frame"""
        try:
            position = robot_pose.pose.position
            orientation = robot_pose.pose.orientation
            
            # Extract yaw
            q = [orientation.x, orientation.y, orientation.z, orientation.w]
            roll, pitch, yaw = tf.euler_from_quaternion(q)
            
            # Robot position in world
            x_robot = position.x
            y_robot = position.y * -1
            
            # Inverse rotation matrix (transpose of rotation matrix)
            R_inv = np.array([
                [np.cos(-yaw), np.sin(-yaw)],
                [-np.sin(-yaw), np.cos(-yaw)]
            ])
            
            robot_points = []
            for pt in world_points:
                # World point
                pt_world = np.array([pt[0], pt[1]])
                
                # Transform: subtract translation, then apply inverse rotation
                pt_relative = pt_world - np.array([x_robot, y_robot])
                pt_robot = R_inv @ pt_relative
                
                robot_points.append([pt_robot[0], pt_robot[1], 0.0])
            
            return robot_points
            
        except Exception as e:
            rospy.logerr(f"World to robot transform error: {e}")
            return []

    def get_merged_frame(self, robot_pose, current_frame):
        """Create a frame with merged current and memorized curves"""
        merged_frame = Frame()
        
        for curve in current_frame.curves:
            if curve.curve_id is None:
                continue
                
            # Start with current frame points
            merged_points = list(curve.points)
            
            # If we have memorized data for this curve ID, add it
            if curve.curve_id in self.memory:
                world_points = self.memory[curve.curve_id]
                # Transform back to robot frame
                robot_points = self.transform_world_to_robot(robot_pose, world_points)
                
                # Merge points
                if robot_points:
                    merged_points.extend(robot_points)
            
            # Filter to only positive x points
            filtered_points = [pt for pt in merged_points if pt[0] > 0]
            
            if len(filtered_points) > 0:
                merged_curve = Curve(filtered_points)
                merged_curve.curve_id = curve.curve_id
                merged_curve.label_noX = curve.label_noX
                merged_curve.label_noY = curve.label_noY
                merged_frame.add_curve(merged_curve)
        
        return merged_frame

    def store_center_line(self, robot_pose, center_line):
        """Store center line in world frame"""
        if not center_line or robot_pose is None:
            return
        
        try:
            position = robot_pose.pose.position
            orientation = robot_pose.pose.orientation
            
            q = [orientation.x, orientation.y, orientation.z, orientation.w]
            roll, pitch, yaw = tf.euler_from_quaternion(q)
            
            x_robot = position.x
            y_robot = position.y * -1
            
            R = np.array([
                [np.cos(-yaw), -np.sin(-yaw)],
                [np.sin(-yaw),  np.cos(-yaw)]
            ])
            
            world_center_line = []
            for pt in center_line:
                pt_robot = np.array([pt[0], pt[1]])
                pt_world = R @ pt_robot + np.array([x_robot, y_robot])
                world_center_line.append([pt_world[0], pt_world[1], 0.0])
            
            self.center_line_memory = world_center_line
            
        except Exception as e:
            rospy.logerr(f"Center line storage error: {e}")

    def get_previous_center_line(self, robot_pose):
        """Get previous center line transformed to current robot frame"""
        if not self.center_line_memory or robot_pose is None:
            return []
        
        return self.transform_world_to_robot(robot_pose, self.center_line_memory)

    def generate_center_from_single_curve(self, robot_pose, single_curve, previous_center_line):
        """
        Generate center line when only one curve is detected
        Uses previous center line and detected curve to estimate center
        """
        # Fix: Properly check numpy array
        if single_curve.points is None or len(single_curve.points) == 0 or len(single_curve.points) < 3:
            return []
        
        # Get previous center line in robot frame
        prev_center = self.get_previous_center_line(robot_pose)
        
        if not prev_center or len(prev_center) < 3:
            # No previous center line, estimate based on lane width
            return self.estimate_center_from_curve(single_curve)
        
        # Determine if curve is on left or right side
        # Compare curve's average y position with previous center line
        curve_y_avg = np.mean([pt[1] for pt in single_curve.points])
        prev_center_y_avg = np.mean([pt[1] for pt in prev_center])
        
        is_left_curve = curve_y_avg < prev_center_y_avg
        
        # Generate opposite side curve by offsetting
        offset_direction = 1 if is_left_curve else -1
        opposite_curve_points = []
        
        for pt in single_curve.points:
            # Offset by lane width in y direction
            opposite_pt = [pt[0], pt[1] + offset_direction * self.lane_width, pt[2]]
            opposite_curve_points.append(opposite_pt)
        
        # Create curves for left and right
        if is_left_curve:
            left_pts = single_curve.points
            right_pts = opposite_curve_points
        else:
            left_pts = opposite_curve_points
            right_pts = single_curve.points
        
        # Calculate center line
        center_line = []
        num_samples = min(len(left_pts), len(right_pts))
        
        for i in range(num_samples):
            lp = left_pts[i] if i < len(left_pts) else left_pts[-1]
            rp = right_pts[i] if i < len(right_pts) else right_pts[-1]
            center_pt = [(lp[0] + rp[0]) / 2.0, (lp[1] + rp[1]) / 2.0, (lp[2] + rp[2]) / 2.0]
            
            # Only add points with positive x
            if center_pt[0] > 0:
                center_line.append(center_pt)
        
        # Blend with previous center line for smoothness
        if len(center_line) > 0 and len(prev_center) > 0:
            center_line = self.blend_center_lines(center_line, prev_center, blend_ratio=0.7)
        
        return center_line

    def estimate_center_from_curve(self, single_curve):
        """Fallback: estimate center by offsetting curve by half lane width"""
        # Fix: Properly check numpy array
        if single_curve.points is None or len(single_curve.points) == 0:
            return []
        
        center_points = []
        half_width = self.lane_width / 2.0
        
        for pt in single_curve.points:
            # Assume curve is on left, offset right to get center
            center_pt = [pt[0], pt[1] + half_width, pt[2]]
            if center_pt[0] > 0:
                center_points.append(center_pt)
        
        return center_points

    def blend_center_lines(self, new_center, prev_center, blend_ratio=0.7):
        """Blend new and previous center lines for smooth transition"""
        if len(prev_center) < 3:
            return new_center
        
        # Interpolate previous center line to match new length
        prev_x = [pt[0] for pt in prev_center]
        prev_y = [pt[1] for pt in prev_center]
        
        new_x = [pt[0] for pt in new_center]
        
        # Interpolate previous y values at new x positions
        prev_y_interp = np.interp(new_x, prev_x, prev_y, left=prev_y[0], right=prev_y[-1])
        
        blended = []
        for i, pt in enumerate(new_center):
            blended_y = blend_ratio * pt[1] + (1 - blend_ratio) * prev_y_interp[i]
            blended.append([pt[0], blended_y, pt[2]])
        
        return blended


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


# ROS publish functions ------------------------------------------------------

def line_marker_publish( current_frame):
    lineMarker_array = MarkerArray()

    clear_marker = Marker()
    clear_marker.action = Marker.DELETEALL
    lineMarker_array.markers.append(clear_marker)

    if not current_frame:
        return lineMarker_array

    for curve in current_frame.curves:
        # === Line Marker ===
        line_marker = Marker()
        line_marker.header.frame_id = "robot_link"
        line_marker.header.stamp = rospy.Time.now()
        line_marker.ns = "curve_lines"
        line_marker.id = curve.curve_id
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.05  # Line thickness

        # Line color: white
        line_marker.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)

        # Add points to line
        for pt in curve.points:
            p = Point(x=pt[0], y= - pt[1], z=0.0)
            line_marker.points.append(p)

        lineMarker_array.markers.append(line_marker)

        # === Text Marker ===
        text_marker = Marker()
        text_marker.header.frame_id = "robot_link"
        text_marker.header.stamp = rospy.Time.now()
        text_marker.ns = "curve_labels"
        text_marker.id = 1000 + curve.curve_id  # Unique ID
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.scale.z = 0.3  # Text height

        # Text color: white
        text_marker.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)

        text_marker.text = f"id {curve.curve_id}, {curve.label_noY}"
        text_marker.pose.position.x = curve.points[-1][0]
        text_marker.pose.position.y = -curve.points[-1][1]
        text_marker.pose.position.z = 0.1
        text_marker.pose.orientation.w = 1.0

        lineMarker_array.markers.append(text_marker)

    return lineMarker_array

def ref_marker_publish( sample_center_line):
    sampleMarker_array = MarkerArray()
    
    if not sample_center_line:
        return sampleMarker_array
    
    for i, point in enumerate(sample_center_line):
        x, y, theta = point
        
        # === Arrow Marker ===
        arrow_marker = Marker()
        arrow_marker.header.frame_id = "robot_link"
        arrow_marker.header.stamp = rospy.Time.now()
        arrow_marker.ns = "sample_center_line"
        arrow_marker.id = i  # Unique ID for each arrow
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD
        
        # Arrow scale (length, width, height)
        arrow_marker.scale.x = 0.2  # Arrow length
        arrow_marker.scale.y = 0.03  # Arrow width
        arrow_marker.scale.z = 0.03  # Arrow height
        
        # Arrow color: blue for sample center line
        arrow_marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)
        
        # Set arrow position
        arrow_marker.pose.position.x = x
        arrow_marker.pose.position.y = -y
        arrow_marker.pose.position.z = 0.0
        
        # Convert theta to quaternion for orientation
        adjusted_theta = -theta
        quaternion = tf.quaternion_from_euler(0, 0, adjusted_theta)
        arrow_marker.pose.orientation.x = quaternion[0]
        arrow_marker.pose.orientation.y = quaternion[1]
        arrow_marker.pose.orientation.z = quaternion[2]
        arrow_marker.pose.orientation.w = quaternion[3]
        
        sampleMarker_array.markers.append(arrow_marker)
    
    return sampleMarker_array

def ref_path_publish( sample_center_line):
    refPath_msg = Float32MultiArray()
    if sample_center_line:
        flattened_points= [coord for point in sample_center_line for coord in point]
        refPath_msg.data   = flattened_points

    return refPath_msg


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
    rospy.loginfo("  /wheeler/ref_path/points - std_msgs.msg/Float32MultiArray")
    rospy.loginfo("  /wheeler/ref_markers - visualization_msgs.msg/MarkerArray")
    rospy.loginfo("  /wheeler/line_markers - visualization_msgs.msg/MarkerArray")

    sub_mask = rospy.Subscriber('/wheeler/lane_node/line_mask', Image, callback = mask_callback)
    sub_depth = rospy.Subscriber('/zed2i/zed_node/depth/depth_registered', Image, callback = depth_callback)
    sub_pose = rospy.Subscriber('/zed2i/zed_node/pose', PoseStamped, callback = pose_callback)
    # sub_camera_info = rospy.Subscriber("/zed2i/zed_node/rgb/camera_info", CameraInfo, callback = camera_info_callback)
    
    refPath_pub = rospy.Publisher('/wheeler/ref_path/points', Float32MultiArray, queue_size=10)
    refMark_pub = rospy.Publisher('/wheeler/ref_markers', MarkerArray, queue_size=10 )
    lineMark_pub = rospy.Publisher('/wheeler/line_markers', MarkerArray, queue_size=10 )


    curve_tracker = CurveTracker(max_frames=5, match_threshold=2000.0)
    memorized_curves = MemorizedCurves(max_distance=50)

    
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
            
            # Get merged frame with current + memorized data
            merged_frame = memorized_curves.get_merged_frame(robot_pose, current_frame)
            merged_frame.classify_curves()
            
            # Calculate center line
            center_line = None
            
            if len(merged_frame.curves) >= 2:
                # Normal case: two or more curves
                center_line = merged_frame.get_center_line()
                rospy.loginfo_throttle(2.0, "Using dual curve center line")
                
            elif len(merged_frame.curves) == 1:
                # Special case: only one curve detected
                single_curve = merged_frame.curves[0]
                prev_center = memorized_curves.get_previous_center_line(robot_pose)
                center_line = memorized_curves.generate_center_from_single_curve(
                    robot_pose, single_curve, prev_center
                )
                rospy.loginfo_throttle(2.0, "Using single curve + previous center line")
                
            else:
                # No curves: use previous center line
                center_line = memorized_curves.get_previous_center_line(robot_pose)
                rospy.loginfo_throttle(2.0, "Using previous center line only")
            
            # Store current center line for next iteration
            if center_line and len(center_line) > 0:
                memorized_curves.store_center_line(robot_pose, center_line)
            
            # Sample center points
            sample_center_line = merged_frame.get_sample_center_points(center_line) if center_line else None


            # white line marker
            lineMark_pub.publish( line_marker_publish(current_frame))
            if sample_center_line:
                refMark_pub.publish(ref_marker_publish(sample_center_line))
                refPath_pub.publish(ref_path_publish(sample_center_line))

            # # Display images
            # cv2.imshow('Line Mask', line_mask)
            # cv2.imshow('Depth', depth_data / np.max(depth_data) if np.max(depth_data) > 0 else depth_data)
            
            # # Display separated lines
            # for i, line_img in enumerate(separated_lines[:5]):  # Limit to 5 windows
            #     cv2.imshow(f"Line {i}", line_img)
            
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