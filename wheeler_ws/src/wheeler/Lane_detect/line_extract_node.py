#!/usr/bin/env python3

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

# Conclude setting / general reprocessing / plots / metrices / datasets
from utils.utils import \
    time_synchronized,select_device, increment_path,\
    scale_coords,xyxy2xywh,non_max_suppression,split_for_trace_model,\
    driving_area_mask,lane_line_mask,plot_one_box,show_seg_result,\
    letterbox,\
    AverageMeter,\
    LoadImages



bridge          = CvBridge()
line_mask       = Image()
depth_data      = Image()
robot_pose      = PoseStamped()


# initialize parameters
shape           = (360, 640)
line_mask_cap   = np.zeros(shape)
rgb_frame_size  = (360, 640)
depth_frame_cap = np.zeros(rgb_frame_size)

rotaion_matrix = np.zeros((3, 3))


t_start     = 0





class Curve:
    def __init__(self, points):
        self.points = points
        self.world_points = []
        self.curve_id = None
        self.x_avg  = 0
        self.y_avg  = 0
        self.label_noX = 0 
        self.label_noY = 0
        self.label_no = (self.label_noX, self.label_noY)
        self.calculate_averages()


    def calculate_averages(self):
        x_vals = [ p[0] for p in self.points]
        y_vals = [ p[1] for p in self.points]
        self.x_avg = sum(x_vals) / len(x_vals) if x_vals else 0
        self.y_avg = sum(y_vals) / len(y_vals) if y_vals else 0


    def offset_to(self, new_curve):
        avg = ((self.x_avg - new_curve.x_avg)**2 + (self.y_avg - new_curve.y_avg)**2)**0.5
        # print("avg", avg)
        return avg


class Frame:
    def __init__(self):
        self.curves = []


    def add_curve(self, curve):
        if len(curve.points) > 0 :
            self.curves.append(curve)


    def classify_curve(self):
        self.curves.sort( key=lambda c: c.x_avg)
        for idx, curve in enumerate( self.curves):
            curve.label_noX = idx

        self.curves.sort( key=lambda c: c.y_avg)
        for idx, curve in enumerate( self.curves):
            curve.label_noY = idx      


class CurveTraking:
    def __init__(self):
        self.frames  = []
        self.max_frames = 5
        self.new_curveId = 1


    def past_curves(self):
        return [ curve for frame in self.frames for curve in frame.curves]


    def match_curve(self, new_curve, past_curves, used_past_curves, threshold=2000):
        best_match = None
        best_offset = float('inf')
        for past_curve in past_curves:
            if past_curve in used_past_curves:
                continue
            offset = new_curve.offset_to(past_curve)
            if offset < threshold and offset < best_offset:
                best_offset = offset
                best_match = past_curve
        return best_match

    def add_frame(self, new_frame):
        past_curves = self.past_curves()
        used_past_curves = set()

        for new_curve in new_frame.curves:
            match = self.match_curve(new_curve, past_curves, used_past_curves)
            if match:
                new_curve.curve_id = match.curve_id
                used_past_curves.add(match)
            else:
                new_curve.curve_id = self.new_curveId
                self.new_curveId += 1

        self.frames.append(new_frame)
        if len(self.frames) > self.max_frames:
            self.frames.pop(0)


class MemorizedCurves:
    def __init__(self):
        self.memory = {}

    def update_memory(self, robot_pose, frame):
        self._transform_to_world(robot_pose, frame)
        # Update memory with new curves from the current frame
        for curve in frame.curves:
            if curve.curve_id is None:
                continue  # Skip unclassified curves

            if curve.curve_id in self.memory:
                # Merge old and new points
                old_points = self.memory[curve.curve_id]
                combined = self._merge_points(old_points, curve.world_points)
                self.memory[curve.curve_id] = combined
            else:
                # First time seeing this curve
                self.memory[curve.curve_id] = list(curve.world_points)

    def _transform_to_world(self, robot_pose, frame):
        position = robot_pose.pose.position
        orientation = robot_pose.pose.orientation

        q = [orientation.x, orientation.y, orientation.z, orientation.w]
        T = tf.quaternion_matrix(q)
        T[0:3, 3] = [position.x, position.y, position.z]

        for curve in frame.curves:
            for pt in curve.points:
                # robot frame convert to camera frame
                pt_home         = np.append( [-1*pt[1], -1*pt[2], pt[0]], 1.0)
                pt_world        = T @ pt_home
                pt_world_coord  = pt_world[:3]
                # camera frame convert to robot frame
                curve.world_points.append( [pt_world_coord[2], -1*pt_world_coord[0], -1*pt_world_coord[1]])

    def _merge_points(self, old_points, new_points):
        # Merge old and new curve points, removing duplicates.
        old_set = {tuple(p) for p in old_points}
        new_set = {tuple(p) for p in new_points}
        merged = list(old_set.union(new_set))
        return merged

    def get_all_curves(self):
        return self.memory




# subscribe camera info
def camera_info_topic(msg):
    global rgb_frame_size
    rgb_frame_size = (msg.height, msg.width)

# subscribe mask frame
def mask_topic(msg):
    global line_mask
    global line_mask_cap
    line_mask      = msg
    line_mask_cap  = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")

# subscribe depth frame
def deth_topic(msg):
    global depth_data
    global depth_frame_cap
    depth_data      = msg
    depth_frame_cap = bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")

# subscribe pose
def pose_topic(msg):
    global rotaion_matrix
    global robot_pose

    robot_pose = msg
    tx, ty, tz = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
    qx, qy, qz, qw = msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w 
    rotation = tf.quaternion_matrix([qx, qy, qz, qw])
    rotaion_matrix = rotation[:3, :3]




# line seperate fn ( according to x axis)
def separate_line_masks(binary_mask):
    contour_areaLimit = 500

    binary_mask = binary_mask.astype(np.uint8)
    gpu_binary_mask = cv2.cuda_GpuMat()
    gpu_binary_mask.upload(binary_mask)

    _, gpu_binary_mask = cv2.cuda.threshold(gpu_binary_mask, 1, 255, cv2.THRESH_BINARY)

    binary_mask = gpu_binary_mask.download()
    contours, _ = cv2.findContours( binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    separated_lines = []
    for i, contour in enumerate(contours):
        # filter noise
        if cv2.contourArea(contour) > contour_areaLimit:
            line_image = np.zeros_like(binary_mask)
            cv2.drawContours( line_image, [contour], -1, 255, thickness=cv2.FILLED)

            separated_lines.append(line_image)

    return separated_lines

# find the line point of line mask
def linemask_2_line(separated_lines):
    # points top to bottom, left to right
    No_oflines = 0
    lines = []
    point_list = []

    for i, separated_line in enumerate( separated_lines):
        No_oflines = i
        # skeletonize the mask
        skeleton = cv2.ximgproc.thinning(separated_line)
        # recode points
        points = cv2.findNonZero(skeleton)

        point_list.append( points.reshape(-1,2))
        # print(points.shape)
        lines.append( skeleton )


    return No_oflines, lines, point_list

# convert 2D coord to 3D coord
def line2D_to_3D( No_lines, line_list, depth):
    global rotaion_matrix

    fx = 959.5588989257812 / 2
    fy = 959.5588989257812 / 2
    cx = 631.89208984375 / 2
    cy = 376.25347900390625 / 2

    list_3Dlines = []
    depth_resized = cv2.cuda_GpuMat()
    depth_resized.upload(depth)

    depth_resized = cv2.cuda.resize(depth_resized, (640, 360), interpolation=cv2.INTER_LINEAR)
    depth_resized_cpu = depth_resized.download()

    for line in line_list:
        point3D_list = []
        for point in line :
            depth = depth_resized_cpu[ point[1], point[0]]

            if depth > 0:            
                x = (point[0] - cx) * depth / fx
                y = (point[1] - cy) * depth / fy
                z = depth
                
                # converting to robot frame
                robot_3dpoint   = np.array([z,-x,-y])
                if ( np.isnan(robot_3dpoint).any() or np.isinf(robot_3dpoint).any()):
                    robot_3dpoint = 0
                else:
                    point3D_list.append( robot_3dpoint)
        list_3Dlines.append(point3D_list)

    return list_3Dlines




if __name__ == '__main__':
    rospy.init_node('line_extract_node')

    rospy.loginfo("line extract node running")
    rospy.loginfo("Published topics:")

    sub_mask        = rospy.Subscriber("/robocop/lane_node/line_mask", Image, callback=mask_topic)
    sub_depth       = rospy.Subscriber("/zed2i/zed_node/depth/depth_registered", Image, callback=deth_topic)
    sub_pose        = rospy.Subscriber("/zed2i/zed_node/pose", PoseStamped, callback=pose_topic)

    rate = rospy.Rate(30)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    curve_traker = CurveTraking()
    memorized_curves = MemorizedCurves()

    frame_count = 0 

    while not rospy.is_shutdown():
        t_start = time.time()

        try:
            frame_count += 1
                
            # # extract line ,ask
            separated_lines = separate_line_masks(line_mask_cap)

            No_lines, lines, point_list = linemask_2_line(separated_lines)
            
            lines_list3D = line2D_to_3D( No_lines, point_list, depth_frame_cap)


            # create frame object
            frame = Frame()
            for line_list in lines_list3D:
                curve = Curve( line_list)
                frame.add_curve(curve)
            # claassified the curve with past curves
            frame.classify_curve()
            curve_traker.add_frame(frame)

            memorized_curves.update_memory( robot_pose, frame)


            # # plot x,y,z 3D ----------------------------------
            # ax.clear()
            # for curve in frame.curves:
            #     # print("curve ids", curve.curve_id)
            #     x = [point[0] for point in curve.points]
            #     y = [point[1] for point in curve.points]
            #     z = 0

            #     ax.scatter(x,y,z)
            #     ax.text(x[-1], y[-1], 0, f"id {curve.curve_id}")
            # ax.set_xlabel('X (Forward)')
            # ax.set_ylabel('Y (left)')
            # ax.set_zlabel('z (Height)')
            # plt.grid(True)
            # plt.draw()


            # # Display the image ----------------------------------------------------
            # cv2.imshow('ll_seg_mask_normal', line_mask_cap)
            # cv2.imshow('depth', depth_frame_cap)


            # # # plot x,y,z 3D ----------------------------------
            # ax.clear()
            # for curve in frame.curves:
            #     # print("curve ids", curve.curve_id)
            #     x = [point[0] for point in curve.world_points]
            #     y = [point[1] for point in curve.world_points]
            #     z = 0

            #     ax.scatter(x,y,z)
            #     ax.text(x[-1], y[-1], 0, f"id {curve.curve_id}")
            # ax.set_xlabel('X (Forward)')
            # ax.set_ylabel('Y (left)')
            # ax.set_zlabel('z (Height)')
            # plt.grid(True)
            # plt.draw()

            # # plot x,y,z 3D ----------------------------------
            ax.clear()
            for curve in frame.curves:
                # print("curve ids", curve.curve_id)
                x = [point[0] for point in curve.world_points]
                y = [point[1] for point in curve.world_points]
                z = 0

                ax.scatter(x,y,z)
                ax.text(x[-1], y[-1], 0, f"id {curve.curve_id}")
            ax.set_xlabel('X (Forward)')
            ax.set_ylabel('Y (left)')
            ax.set_zlabel('z (Height)')
            plt.grid(True)
            plt.draw()


            # Display the image ----------------------------------------------------
            cv2.imshow('ll_seg_mask_normal', line_mask_cap)
            cv2.imshow('depth', depth_frame_cap)


            for i, line_img in enumerate(lines):
                cv2.imshow("Line "+str(i), line_img)


            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        except Exception as e:
            rospy.logerr(f"Error {e}")
            traceback.print_exc()



        print(" Point extract FPS : ", 1/(time.time()-t_start) )

        # # memory clean
        # del img, pred, anchor_grid, ll_seg_mask
        # torch.cuda.empty_cache()
        # gc.collect()

        rate.sleep()
    
    plt.ioff()
    cv2.destroyAllWindows()
    rospy.spin()

