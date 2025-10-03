#!/usr/bin/env python3


import rospy
import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np
import struct
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField
from cv_bridge import CvBridge
import matplotlib.pyplot as plt

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
fx, fy, cx, cy = 0, 0, 0, 0

fpsCount    = 0
t_start     = 0
weights     = 'data/weights/yolopv2.pt'
conf_thres  = 0.3
iou_thres   = 0.45

inf_time    = AverageMeter()
waste_time  = AverageMeter()
nms_time    = AverageMeter()



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





# subscribe camera info
def camera_info_topic(msg):
    global rgb_frame_size
    global fx, fy, cx, cy

    rgb_frame_size = (msg.height, msg.width)
    fx = msg.K[0]
    fy = msg.K[4]
    cx = msg.K[2]
    cy = msg.K[5]

    # print("rgb_frame_size", rgb_frame_size)
    # print("fx, fy, cx, cy", fx, fy, cx, cy)



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
    line_points = np.empty([0,2])

    for i, separated_line in enumerate( separated_lines):
        lines = i
        # skeletonize the mask
        skeleton = cv2.ximgproc.thinning(separated_line)

        # find the contours on the skeleton
        contours, _ = cv2.findContours( skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        points = np.vstack(contours).squeeze()
        line_points = np.vstack((line_points, points))

    return line_points


# create 3d coordinate
def line_3d_poins(pixel_points, depth_frame):
    global fx, fy, cx, cy

    line_points_3d = []
    for u, v in pixel_points:
        # print("u,v", u,v)
        Z = depth_frame[ int(v), int(u)]
        if Z > 0:
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            line_points_3d.append((X,Y,Z))

    return line_points_3d


# 3d coordinate to point cloud
def create_pointCloud( coordinate_points):
    header = rospy.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "map"  # Use different frames for each lane (e.g., "lane_1_frame")

    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1)
    ]

    cloud_data = []
    for point in coordinate_points:
        cloud_data.append(struct.pack("fff", *point))  # Convert to binary

    return PointCloud2(
        header=header,
        height=1,
        width=len(coordinate_points),
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=12,  # 3x float32 (4 bytes each)
        row_step=12 * len(coordinate_points),
        data=b"".join(cloud_data)
    )



if __name__ == '__main__':
    rospy.init_node('laneDetect_node')
    rospy.loginfo("lane detect node start")

    sub_rgb_camInfo = rospy.Subscriber("/zed2i/zed_node/rgb/camera_info", CameraInfo, callback=camera_info_topic)
    sub_rgb_img     = rospy.Subscriber("/zed2i/zed_node/rgb/image_rect_color", Image, callback=rgb_topic)
    sub_depth       = rospy.Subscriber("/zed2i/zed_node/depth/depth_registered", Image, callback=deth_topic)
    
    pub_line_cloud = rospy.Publisher("/line_cloud", PointCloud2, queue_size=1)

    rate = rospy.Rate(15)

    while not rospy.is_shutdown():

        t_start = time.time()

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


        # Inference
        t1 = time_synchronized()
        [pred,anchor_grid],seg,ll= model(img)
        t2 = time_synchronized()

        # waste time: the incompatibility of  torch.jit.trace causes extra time consumption in demo version 
        # but this problem will not appear in offical version 
        tw1 = time_synchronized()
        pred = split_for_trace_model(pred,anchor_grid)
        tw2 = time_synchronized()

        # Apply NMS
        t3 = time_synchronized()
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        t4 = time_synchronized()
   
        # get the mask of drive lane and lane line
        ll_seg_mask = lane_line_mask(ll)

        # resize frames to (640,360) and create 3d mask
        ll_seg_mask = cv2.resize(ll_seg_mask, (640, 360), interpolation=cv2.INTER_NEAREST)
        ll_seg_mask = (ll_seg_mask*255).astype(np.uint8)




        # separate lines
        separated_lines = separate_line_masks(ll_seg_mask)
        # extract middle points
        points = linemask_2_line(separated_lines)

        # generate 3d point
        line_points_3d = line_3d_poins( points, depth_frame_cap)
        print(line_points_3d)

        # generate cloud msg
        cloud_msg = create_pointCloud(line_points_3d)

        pub_line_cloud.publish(cloud_msg)


        print("displaying fps : ", 1/(time.time()-t_start) )

        



        try:

            # Display the image
            cv2.imshow("Camera Image", rgb_frame_cap)
            cv2.imshow('ll_seg_mask_normal', ll_seg_mask)

            for i, line_img in enumerate(separated_lines):
                cv2.imshow("Line "+str(i), line_img)

            # if ( len(points) > 0):
            #     plt.plot( points[:,0], -points[:,1], 'o', label="curve")
            #     plt.show()

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        except Exception as e:
            rospy.logerr(f"Failed to display frame {e}")


        


        rate.sleep()
    

    rospy.spin()

