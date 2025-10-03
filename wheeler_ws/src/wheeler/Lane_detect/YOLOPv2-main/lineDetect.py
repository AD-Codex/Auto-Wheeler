#!/usr/bin/env python3


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



bridge      = CvBridge()
rgba_img    = Image()
depth_data  = Image()


# initialize parameters
shape           = (360, 640, 3)
rgb_frame_cap   = np.zeros(shape)
rgb_frame_size  = (360, 640)
depth_frame_cap = np.zeros(rgb_frame_size)

rotaion_matrix = np.zeros((3, 3))


# fpsCount    = 0
t_start     = 0
# weights     = 'data/weights/yolopv2.pt'
conf_thres  = 0.3
iou_thres   = 0.45

# inf_time    = AverageMeter()
# waste_time  = AverageMeter()
# nms_time    = AverageMeter()



# Load model
stride  = 32
model   = torch.jit.load('/home/jetson/ROS/ROBOT2/src/robocop_control/Lane_detect/YOLOPv2-main/yolopv2.pt')
device  = select_device('cuda')  # cuda device, i.e. 0 or 0,1,2,3 or cpu
half    = device.type != 'cpu'  # half precision only supported on CUDA
model   = model.to(device)

if half:
    model.half()  # to FP16  
model.eval()


# Run inference
if device.type != 'cpu':
    model(torch.zeros(1, 3, 640, 640).to(device).type_as(next(model.parameters())))  # run once
t0 = time.time()



class Model:
    def __init__(self):
        self.conf_thres  = 0.3
        self.iou_thres   = 0.45

        # Load model
        self.stride  = 32
        self.model   = torch.jit.load('/home/jetson/ROS/ROBOT2/src/robocop_control/Lane_detect/YOLOPv2-main/yolopv2.pt')
        self.device  = select_device('cuda')  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.half    = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model   = self.model.to(self.device)

        if self.half:
            self.model.half()  # to FP16  
        self.model.eval()


        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, 640, 640).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.t0 = time.time()


    def convert_blackWhite(self, mask):
        converted_mask = cv2.resize(mask, (1280, 720), interpolation=cv2.INTER_NEAREST)
        converted_mask = (converted_mask*255).astype(np.uint8)
        return converted_mask


    def detect(self, rgb_frame_cap):
        img0 = cv2.resize(rgb_frame_cap, (1280,720), interpolation=cv2.INTER_LINEAR)
        img = letterbox(img0, 640, stride=32)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to( self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            # Inference
            [ pred, anchor_grid], seg, ll = self.model(img)
            pred = split_for_trace_model( pred, anchor_grid)
            # Apply NMS
            pred = non_max_suppression( pred, conf_thres, iou_thres)
           
        # get the mask of drive lane and lane line
        ll_seg_mask = lane_line_mask(ll)

        # reshape to 1280 x 720 and mask to 0 and 255
        ll_seg_mask = self.convert_blackWhite(ll_seg_mask)

        return ll_seg_mask



class Curve:
    def __init__(self, points):
        self.points = points
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
        avg = ((self.x_avg - new_curve.x_avg)**2 - (self.y_avg - new_curve.y_avg)**2)**0.5
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

    def match_curve(self, new_curve, past_curves, threshold=2000):
        matching_curve = None
        for past_curve in past_curves:
            offset = new_curve.offset_to(past_curve)
            # print(offset)
            if offset < threshold:
                matching_curve = past_curve
        return matching_curve

    def add_frame(self, new_frame):
        past_curves = self.past_curves()

        for new_curve in new_frame.curves:
            match_curve = self.match_curve(new_curve, past_curves)
            if match_curve:
                new_curve.curve_id = match_curve.curve_id
            else:
                new_curve.curve_id = self.new_curveId
                self.new_curveId += 1

        self.frames.append(new_frame)
        if len(self.frames) > self.max_frames:
            self.frames.pop(0)

            


# subscribe camera info
def camera_info_topic(msg):
    global rgb_frame_size
    rgb_frame_size = (msg.height, msg.width)


# subscribe rgb frame
def rgb_topic(msg):
    global rgba_img
    global rgb_frame_cap
    rgba_img      = msg
    rgb_frame_cap = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")


# subscribe depth frame
def deth_topic(msg):
    global depth_data
    global depth_frame_cap
    depth_data      = msg
    depth_frame_cap = bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")


# subscribe pose
def pose_topic(msg):
    global rotaion_matrix
    tx, ty, tz = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
    qx, qy, qz, qw = msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w 
    rotation = tf.quaternion_matrix([qx, qy, qz, qw])
    rotaion_matrix = rotation[:3, :3]

# line seperate fn ( according to x axis)
def separate_line_masks(binary_mask):
    contour_areaLimit = 500

    binary_mask = binary_mask.astype(np.uint8)
    _, binary_mask = cv2.threshold( binary_mask, 1, 255, cv2.THRESH_BINARY)
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

    fx = 959.5588989257812
    fy = 959.5588989257812
    cx = 631.89208984375
    cy = 376.25347900390625

    list_3Dlines = []

    depth_coord = cv2.resize( depth, (1280, 720), interpolation=cv2.INTER_LINEAR)
    
    for line in line_list:
        point3D_list = []
        for point in line :
            depth = depth_coord[ point[1], point[0]]

            if depth > 0:            
                x = (point[0] - cx) * depth / fx
                y = (point[1] - cy) * depth / fy
                z = depth
                
                world_3dpoint = np.array(rotaion_matrix) @ np.array([z,-x,-y])
                # print(world_3dpoint)
                if ( np.isnan(world_3dpoint).any() or np.isinf(world_3dpoint).any()):
                    world_3dpoint = 0
                else:
                    point3D_list.append( world_3dpoint)
        list_3Dlines.append(point3D_list)

    return list_3Dlines





if __name__ == '__main__':
    rospy.init_node('laneDetect_node')
    rospy.loginfo("lane detect node start")

    # sub_rgb_camInfo = rospy.Subscriber("/zed2i/zed_node/rgb/camera_info", CameraInfo, callback=camera_info_topic)
    sub_rgb_img     = rospy.Subscriber("/zed2i/zed_node/rgb/image_rect_color", Image, callback=rgb_topic)
    sub_depth       = rospy.Subscriber("/zed2i/zed_node/depth/depth_registered", Image, callback=deth_topic)
    sub_pose        = rospy.Subscriber("/zed2i/zed_node/pose", PoseStamped, callback=pose_topic)

    rate = rospy.Rate(30)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    curve_traker = CurveTraking()

    while not rospy.is_shutdown():
        t_start = time.time()

        try:
            img0 = cv2.resize(rgb_frame_cap, (1280,720), interpolation=cv2.INTER_LINEAR)
            img = letterbox(img0, 640, stride=32)[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            with torch.no_grad():
                # Inference
                [pred,anchor_grid],seg,ll= model(img)
                pred = split_for_trace_model(pred,anchor_grid)
                # Apply NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres)
           
            # get the mask of drive lane and lane line
            ll_seg_mask = lane_line_mask(ll)

            # resize frames to (640,360) and create 3d mask
            ll_seg_mask = cv2.resize(ll_seg_mask, (1280, 720), interpolation=cv2.INTER_NEAREST)
            ll_seg_mask = (ll_seg_mask*255).astype(np.uint8)

            separated_lines = separate_line_masks(ll_seg_mask)

            No_lines, lines, point_list = linemask_2_line(separated_lines)
            
            lines_list3D = line2D_to_3D( No_lines, point_list, depth_frame_cap)


            frame = Frame()
            for line_list in lines_list3D:
                curve = Curve( line_list)
                frame.add_curve(curve)
            frame.classify_curve()
            curve_traker.add_frame(frame)

            # plot x,y,z 3D
            ax.clear()
            for curve in frame.curves:
                print("curve ids", curve.curve_id)
                x = [point[0] for point in curve.points]
                y = [point[1] for point in curve.points]
                z = 0

                ax.scatter(x,y,z)
                ax.text(x[-1], y[-1], 0, f"id {curve.curve_id}")
            ax.set_xlabel('X (Forward)')
            ax.set_ylabel('Y (left)')
            ax.set_zlabel('z (Height)')
            plt.grid(True)
            plt.draw()




            # Display the image ----------------------------------------------------
            cv2.imshow("Camera Image", rgb_frame_cap)
            cv2.imshow('ll_seg_mask_normal', ll_seg_mask)
            cv2.imshow('depth', depth_frame_cap)

            for i, line_img in enumerate(lines):
                cv2.imshow("Line "+str(i), line_img)

            

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        except Exception as e:
            rospy.logerr(f"Error {e}")
            traceback.print_exc()



        print(" fps : ", 1/(time.time()-t_start) )

        # # memory clean
        # del img, pred, anchor_grid, ll_seg_mask
        # torch.cuda.empty_cache()
        # gc.collect()

        rate.sleep()
    
    plt.ioff()
    cv2.destroyAllWindows()
    rospy.spin()

