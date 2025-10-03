#!/usr/bin/env python3


import rospy
import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

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



if __name__ == '__main__':
    rospy.init_node('laneDetect_node')
    rospy.loginfo("lane detect node start")

    sub_rgb_camInfo = rospy.Subscriber("/zed2i/zed_node/rgb/camera_info", CameraInfo, callback=camera_info_topic)
    sub_rgb_img     = rospy.Subscriber("/zed2i/zed_node/rgb/image_rect_color", Image, callback=rgb_topic)
    sub_depth       = rospy.Subscriber("/zed2i/zed_node/depth/depth_registered", Image, callback=deth_topic)

    pub_drive_mask  = rospy.Publisher('/robocop/laneDetect_node/lane_drive', Image, queue_size=20)
    pub_line_mask   = rospy.Publisher('/robocop/laneDetect_node/lane_line', Image, queue_size=20)

    
    rate = rospy.Rate(10)

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
        da_seg_mask = driving_area_mask(seg)



        # resize frames to (640,360) and create 3d mask
        ll_seg_mask = cv2.resize(ll_seg_mask, (640, 360), interpolation=cv2.INTER_NEAREST)
        ll_seg_mask = (ll_seg_mask*255).astype(np.uint8)

        da_seg_mask = cv2.resize(da_seg_mask, (640, 360), interpolation=cv2.INTER_NEAREST)
        da_seg_mask = (da_seg_mask*255).astype(np.uint8)


        # publish drive lane and lane line
        lane_line_img = bridge.cv2_to_imgmsg(ll_seg_mask, encoding="mono8")
        pub_line_mask.publish(lane_line_img)

        lane_drive_img = bridge.cv2_to_imgmsg(da_seg_mask, encoding="mono8")
        pub_drive_mask.publish(lane_drive_img)


        print("displaying fps : ", 1/(time.time()-t_start) )

        try:
            # Display the image
            cv2.imshow("Camera Image", rgb_frame_cap)
            cv2.imshow('ll_seg_mask_normal', ll_seg_mask)
            cv2.imshow('da_seg_mask_normal', da_seg_mask)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        except Exception as e:
            rospy.logerr(f"Failed to display frame {e}")


        rate.sleep()
    

    rospy.spin()

