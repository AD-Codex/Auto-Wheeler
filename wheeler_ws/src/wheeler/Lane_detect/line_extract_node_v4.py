#!/usr/bin/env python3

# World coordinate lane detection with memorized curves - Marker visualization

import rospy
import time
import cv2
import numpy as np
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from cv_bridge import CvBridge
import tf.transformations as tf
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
            
        self.curves.sort(key=lambda c: c.x_avg)
        for idx, curve in enumerate(self.curves):
            curve.label_noX = idx

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
    def __init__(self, max_distance=2.0):
        self.memory = {}  # curve_id -> list of world points
        self.max_distance = max_distance

    def update_memory(self, robot_pose, frame):
        if robot_pose is None:
            rospy.logwarn("No robot pose available for memory update")
            return
            
        # Transform current frame curves to world coordinates
        self.transform_to_world(robot_pose, frame)
        
        # Update memory with new curves
        for curve in frame.curves:
            if curve.curve_id is None or not curve.world_points:
                continue

            if curve.curve_id in self.memory:
                # Merge with existing curve (no time-based removal)
                old_points = self.memory[curve.curve_id]
                merged_points = self.merge_points_weighted(old_points, curve.world_points)
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

    def merge_points_weighted(self, old_points, new_points, 
                            new_weight=0.7, old_weight=0.3, 
                            distance_threshold=0.5):
        """Merge old and new curve points with weighted averaging"""
        if not old_points:
            return list(new_points)
        if not new_points:
            return list(old_points)

        old_array = np.array([[pt[0], pt[1]] for pt in old_points])
        new_array = np.array([[pt[0], pt[1]] for pt in new_points])
        
        distances = cdist(new_array, old_array)
        
        merged_points = []
        used_old_indices = set()
        
        # Process new points
        for i, new_pt in enumerate(new_points):
            min_dist_idx = np.argmin(distances[i])
            min_dist = distances[i][min_dist_idx]
            
            if min_dist < distance_threshold and min_dist_idx not in used_old_indices:
                # Weighted average merge
                old_pt = old_points[min_dist_idx]
                merged_pt = [
                    new_weight * new_pt[0] + old_weight * old_pt[0],
                    new_weight * new_pt[1] + old_weight * old_pt[1],
                    new_weight * new_pt[2] + old_weight * old_pt[2]
                ]
                merged_points.append(merged_pt)
                used_old_indices.add(min_dist_idx)
            else:
                merged_points.append(list(new_pt))
        
        # Add unused old points
        for i, old_pt in enumerate(old_points):
            if i not in used_old_indices:
                merged_points.append(list(old_pt))
        
        return merged_points

    # def get_center_line(self):
    #     """Calculate center line from memorized curves"""
    #     if len(self.memory) < 2:
    #         return []

    #     # Get all curves sorted by their average Y position
    #     curve_data = []
    #     for curve_id, points in self.memory.items():
    #         if len(points) > 10:
    #             y_avg = np.mean([pt[1] for pt in points])
    #             curve_data.append((y_avg, points))
        
    #     if len(curve_data) < 2:
    #         return []
        
    #     # Sort and get left/right curves
    #     curve_data.sort(key=lambda x: x[0])
    #     left_points = curve_data[0][1]
    #     right_points = curve_data[-1][1]

    #     left_array = np.array(left_points)
    #     right_array = np.array(right_points)

    #     # Find common X range
    #     x_min = max(np.min(left_array[:, 0]), np.min(right_array[:, 0]))
    #     x_max = min(np.max(left_array[:, 0]), np.max(right_array[:, 0]))
        
    #     if x_max <= x_min:
    #         return []

    #     x_samples = np.arange(x_min, x_max, 0.5)
    #     center_line = []

    #     for x in x_samples:
    #         left_dists = np.abs(left_array[:, 0] - x)
    #         right_dists = np.abs(right_array[:, 0] - x)
            
    #         left_idx = np.argmin(left_dists)
    #         right_idx = np.argmin(right_dists)
            
    #         center_x = x
    #         center_y = (left_array[left_idx, 1] + right_array[right_idx, 1]) / 2.0
            
    #         # Calculate theta
    #         if len(center_line) > 0:
    #             dx = center_x - center_line[-1][0]
    #             dy = center_y - center_line[-1][1]
    #             theta = np.arctan2(dy, dx)
    #         else:
    #             theta = 0.0
            
    #         center_line.append([center_x, center_y, theta])

    #     return center_line

    def get_center_line(self, robot_pose):
        """Calculate center line from memorized curves, considering only points in front of robot"""
        if len(self.memory) < 2 or robot_pose is None:
            return []

        # Get robot pose
        position = robot_pose.pose.position
        orientation = robot_pose.pose.orientation
        
        # Extract yaw
        q = [orientation.x, orientation.y, orientation.z, orientation.w]
        roll, pitch, yaw = tf.euler_from_quaternion(q)
        
        robot_x = position.x
        robot_y = position.y * -1
        
        # Inverse rotation matrix (world to robot)
        R_inv = np.array([
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)]
        ])
        
        # Transform all curves to robot frame and filter points in front
        curves_robot_frame = []
        
        for curve_id, world_points in self.memory.items():
            if len(world_points) < 10:
                continue
                
            robot_frame_points = []
            
            for pt in world_points:
                # Transform world point to robot frame
                pt_world = np.array([pt[0] - robot_x, pt[1] - robot_y])
                pt_robot = R_inv @ pt_world
                
                # Only keep points in front of robot (x > 0) and within reasonable range
                if pt_robot[0] > 0 and pt_robot[0] < 15.0:  # 0 to 15m in front
                    robot_frame_points.append([pt_robot[0], pt_robot[1], pt[2]])
            
            if len(robot_frame_points) > 10:
                # Calculate average Y position in robot frame for left/right sorting
                y_avg = np.mean([p[1] for p in robot_frame_points])
                curves_robot_frame.append((y_avg, robot_frame_points, curve_id))
        
        if len(curves_robot_frame) < 2:
            return []
        
        # Sort by Y position in robot frame (left to right)
        curves_robot_frame.sort(key=lambda x: x[0])
        
        # Get leftmost and rightmost curves
        left_points = curves_robot_frame[0][1]
        right_points = curves_robot_frame[-1][1]
        
        left_array = np.array(left_points)
        right_array = np.array(right_points)
        
        # Find common X range (in robot frame)
        x_min = max(np.min(left_array[:, 0]), np.min(right_array[:, 0]))
        x_max = min(np.max(left_array[:, 0]), np.max(right_array[:, 0]))
        
        if x_max <= x_min:
            return []
        
        # Sample center line in robot frame
        x_samples = np.arange(x_min, x_max, 0.5)
        center_line_robot = []
        
        for x in x_samples:
            left_dists = np.abs(left_array[:, 0] - x)
            right_dists = np.abs(right_array[:, 0] - x)
            
            left_idx = np.argmin(left_dists)
            right_idx = np.argmin(right_dists)
            
            center_x = x
            center_y = (left_array[left_idx, 1] + right_array[right_idx, 1]) / 2.0
            
            center_line_robot.append([center_x, center_y])
        
        if len(center_line_robot) < 2:
            return []
        
        # Transform center line back to world frame
        R = np.array([
            [np.cos(-yaw), -np.sin(-yaw)],
            [np.sin(-yaw), np.cos(-yaw)]
        ])
        
        center_line_world = []
        
        for i, pt_robot in enumerate(center_line_robot):
            # Transform to world
            pt_world = R @ np.array([pt_robot[0], pt_robot[1]]) + np.array([robot_x, robot_y])
            
            # Calculate theta
            if i > 0:
                dx = pt_world[0] - center_line_world[-1][0]
                dy = pt_world[1] - center_line_world[-1][1]
                theta = np.arctan2(dy, dx)
            else:
                theta = 0.0
            
            center_line_world.append([pt_world[0], pt_world[1], theta])
        
        return center_line_world

# ROS callback functions
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
    _, binary_mask = cv2.threshold(binary_mask, 1, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3,3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
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

# Marker publishing functions
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


def publish_world_memorized_markers(memorized_curves):
    """Publish all memorized curves as markers in world frame"""
    marker_array = MarkerArray()
    
    # Clear old markers
    clear_marker = Marker()
    clear_marker.action = Marker.DELETEALL
    marker_array.markers.append(clear_marker)
    
    if not memorized_curves.memory:
        return marker_array
    
    for curve_id, world_points in memorized_curves.memory.items():
        if not world_points or len(world_points) < 3:
            continue
        
        # Line marker with SORTED points
        line_marker = Marker()
        line_marker.header.frame_id = "world_link"
        line_marker.header.stamp = rospy.Time.now()
        line_marker.ns = "memorized_curves"
        line_marker.id = curve_id
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.08
        line_marker.color = ColorRGBA(0.0, 1.0, 1.0, 0.9)  # Cyan

        # Sort points by X coordinate (forward direction) before adding
        sorted_points = sorted(world_points, key=lambda pt: pt[0])

        for pt in sorted_points:
            p = Point(x=pt[0], y=pt[1], z=0.0)
            line_marker.points.append(p)

        marker_array.markers.append(line_marker)
        
        # Text label
        text_marker = Marker()
        text_marker.header.frame_id = "world_link"
        text_marker.header.stamp = rospy.Time.now()
        text_marker.ns = "memorized_labels"
        text_marker.id = 1000 + curve_id
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.scale.z = 0.4
        text_marker.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
        text_marker.text = f"ID:{curve_id}"
        text_marker.pose.position.x = world_points[-1][0]
        text_marker.pose.position.y = world_points[-1][1]
        text_marker.pose.position.z = 0.3
        text_marker.pose.orientation.w = 1.0
        
        marker_array.markers.append(text_marker)
    
    return marker_array

def publish_world_center_line_markers(center_line):
    """Publish center line as arrow markers in world frame"""
    marker_array = MarkerArray()
    
    if not center_line or len(center_line) < 2:
        return marker_array
    
    for i, point in enumerate(center_line):
        x, y, theta = point
        
        arrow_marker = Marker()
        arrow_marker.header.frame_id = "world_link"
        arrow_marker.header.stamp = rospy.Time.now()
        arrow_marker.ns = "world_center_line"
        arrow_marker.id = i
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD
        
        arrow_marker.scale.x = 0.3
        arrow_marker.scale.y = 0.05
        arrow_marker.scale.z = 0.05
        
        arrow_marker.color = ColorRGBA(0.0, 1.0, 0.0, 1.0)  # Green
        
        arrow_marker.pose.position.x = x
        arrow_marker.pose.position.y = y
        arrow_marker.pose.position.z = 0.0
        
        quaternion = tf.quaternion_from_euler(0, 0, theta)
        arrow_marker.pose.orientation.x = quaternion[0]
        arrow_marker.pose.orientation.y = quaternion[1]
        arrow_marker.pose.orientation.z = quaternion[2]
        arrow_marker.pose.orientation.w = quaternion[3]
        
        marker_array.markers.append(arrow_marker)
    
    return marker_array

def publish_reference_path(center_line, pub):
    """Publish reference path"""
    if center_line and len(center_line) > 0:
        msg = Float32MultiArray()
        flattened = []
        for pt in center_line:
            flattened.extend([pt[0], pt[1], pt[2]])
        msg.data = flattened
        pub.publish(msg)




def main():
    rospy.init_node('world_memorized_lane_markers_node')
    rospy.loginfo("World frame memorized lane markers node started")
    rospy.loginfo("Published topics:")
    rospy.loginfo("  /wheeler/world_memorized_markers - Memorized curves (cyan)")
    rospy.loginfo("  /wheeler/world_center_markers - Center line (green arrows)")
    rospy.loginfo("  /wheeler/world_ref_path/points - Reference path data")

    # Subscribers
    sub_mask = rospy.Subscriber('/wheeler/lane_node/line_mask', Image, callback=mask_callback)
    sub_depth = rospy.Subscriber('/zed2i/zed_node/depth/depth_registered', Image, callback=depth_callback)
    sub_pose = rospy.Subscriber('/zed2i/zed_node/pose', PoseStamped, callback=pose_callback)
    
    # Publishers
    refPath_pub = rospy.Publisher('/wheeler/ref_path/points', Float32MultiArray, queue_size=10)
    refMark_pub = rospy.Publisher('/wheeler/ref_markers', MarkerArray, queue_size=10 )
    lineMark_pub = rospy.Publisher('/wheeler/line_markers', MarkerArray, queue_size=10 )

    world_memorized_pub = rospy.Publisher('/wheeler/world_memorized_markers', MarkerArray, queue_size=10)
    world_center_pub = rospy.Publisher('/wheeler/world_center_markers', MarkerArray, queue_size=10)
    refPath_pub = rospy.Publisher('/wheeler/world_ref_path/points', Float32MultiArray, queue_size=10)

    # Initialize components
    curve_tracker = CurveTracker(max_frames=5, match_threshold=2.0)
    memorized_curves = MemorizedCurvesWorldFrame(max_distance=2.0)
    
    rate = rospy.Rate(30)
    frame_count = 0
    
    rospy.loginfo("Starting processing loop")
    
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

            # Process pipeline
            separated_lines = separate_line_masks(line_mask, min_area=300)
            point_lists = extract_line_points(separated_lines, max_points_per_line=50)
            lines_3d = convert_2d_to_3d(point_lists, depth_data)
            
            # Create frame with detected curves
            current_frame = Frame()
            for line_points in lines_3d:
                curve = Curve(line_points)
                current_frame.add_curve(curve)
            
            # Track and memorize curves
            current_frame.classify_curves()
            curve_tracker.add_frame(current_frame)
            memorized_curves.update_memory(robot_pose, current_frame)

            # robot frame center line
            center_line = current_frame.get_center_line()
            sample_center_line = current_frame.get_sample_center_points(center_line)

            # robot frame white line marker
            lineMark_pub.publish( line_marker_publish(current_frame))
            refMark_pub.publish( ref_marker_publish(sample_center_line))
            refPath_pub.publish( ref_path_publish(sample_center_line))

            
            # Calculate world frame center line from memorized curves
            world_center_line = memorized_curves.get_center_line( robot_pose)
            
            # Publish markers
            world_memorized_pub.publish(publish_world_memorized_markers(memorized_curves))
            world_center_pub.publish(publish_world_center_line_markers(world_center_line))
            publish_reference_path(world_center_line, refPath_pub)

            # Performance monitoring
            processing_time = time.time() - start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0
            
            if frame_count % 30 == 0:
                rospy.loginfo(f"FPS: {fps:.1f}, Memorized curves: {len(memorized_curves.memory)}, Current: {len(current_frame.curves)}")
                
        except Exception as e:
            rospy.logerr(f"Processing error: {e}")
            traceback.print_exc()
        
        rate.sleep()
    
    rospy.loginfo("World frame memorized lane markers node shutting down")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass