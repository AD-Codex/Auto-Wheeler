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
from threading import Lock

# Conclude setting / general reprocessing / plots / metrices / datasets
from utils.utils import \
    time_synchronized,select_device, increment_path,\
    scale_coords,xyxy2xywh,non_max_suppression,split_for_trace_model,\
    driving_area_mask,lane_line_mask,plot_one_box,show_seg_result,\
    letterbox,\
    AverageMeter,\
    LoadImages


# Global variables
bridge = CvBridge()
data_lock = Lock()


class ImageData:
    # initialize parameters
    def __init__(self):
        self.shape              = (360, 640, 3)
        self.rgb_frame_cap      = np.zeros(self.shape)
        self.rgb_frame_size     = (360, 640)
        self.depth_frame_cap    = np.zeros(self.rgb_frame_size)

        self.process_every_n_frames = 3
        self.frame_count = 0

image_data = ImageData()


# model calss
class Model:
    def __init__(self):
        # model parameters
        self.conf_thres  = 0.3
        self.iou_thres   = 0.45

        # Load model
        self.stride  = 32
        try:
            self.model   = torch.jit.load('/home/dell/Documents/ROBOT2/src/robocop_control/Lane_detect/YOLOPv2-main/yolopv2.pt')
            rospy.loginfo(f"Model loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to load model: {e}")
        
        # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.device  = select_device('cuda' if torch.cuda.is_available() else 'cpu')  
        # half precision only supported on CUDA
        self.half    = self.device.type != 'cpu' and torch.cuda.is_available()
        
        self.model   = self.model.to(self.device)

        if self.half:
            self.model.half()  # to FP16  
        
        self.model.eval()

        self.runones_model()

        rospy.loginfo(f"model on device: {self.device}")
        rospy.loginfo(f"half precision: {self.half}")


    def runones_model(self):
        if self.device.type != 'cpu':
            try:
                self.model(torch.zeros(1, 3, 640, 640).to(self.device).type_as(next(self.model.parameters())))
                rospy.loginfo("Model run ones completed")
            except Exception as e:
                rospy.logwarn("Model run ones failed: {e}")


    def convert_blackWhite(self, mask):
        try:
            converted_mask = cv2.resize(mask, (640, 360), interpolation=cv2.INTER_NEAREST)
            converted_mask = (converted_mask*255).astype(np.uint8)
            return converted_mask
        except Exception as e:
            rospy.logerr(f"Error in convert_blackWhite: {e}")
            return np.zeros((360, 640), dtype=np.uint8)


    def detect(self, rgb_frame_cap):
        # process image
        try:
            # Resize image
            img0 = cv2.resize(rgb_frame_cap, (1280,720), interpolation=cv2.INTER_LINEAR)
            # Convert
            img = letterbox(img0, 640, stride=32)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            # Convert to tensor
            img = torch.from_numpy(img).to( self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            if img.ndimension() == 3:
                img = img.unsqueeze(0)
        except Exception as e:
            rospy.logerr(f"Error in image preprocessing: {e}")

        # print("Input image device:", img.device)
        # print("Input image dtype:", img.dtype)

        # main detection
        try:
            with torch.no_grad():
                # Inference
                start_time = time_synchronized()
                # [ pred, anchor_grid], seg, ll = self.model(img)
                with torch.cuda.amp.autocast():  # Mixed precision
                    [pred, anchor_grid], seg, ll = self.model(img)
                torch.cuda.synchronize()
                inference_time = time.time() - start_time
                print(f"Inference time: {inference_time:.3f} s | FPS: {1/inference_time:.2f}")

                # pred = split_for_trace_model( pred, anchor_grid)
                # # Apply NMS
                # pred = non_max_suppression( pred, self.conf_thres, self.iou_thres)
           
            # get the mask of drive lane and lane line
            ll_seg_mask = lane_line_mask(ll)

            # reshape to 640 x 360 and mask to 0 and 255
            ll_seg_mask = self.convert_blackWhite(ll_seg_mask)

            return ll_seg_mask
        except Exception as e:
            rospy.logerr(f"Error in detection: {e}")
            return np.zeros((360, 640), dtype=np.uint8)


# ROs callback fn --------------------------------------------------------------
def image_callback(msg):
    with data_lock:
        image_data.rgb_frame_cap = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")


def main():
    rospy.init_node('line_mask_node')

    rospy.loginfo("line mask node running")
    rospy.loginfo("Published topics:")
    rospy.loginfo("  /wheeler/lane_node/line_mask - sensor_msgs.msg/Image")

    sub_rgb_img     = rospy.Subscriber('/zed2i/zed_node/rgb/image_rect_color', Image, callback=image_callback)
    
    pub_line_mask   = rospy.Publisher('/wheeler/lane_node/line_mask', Image, queue_size=20)

    model = Model()

    rate = rospy.Rate(60)

    while not rospy.is_shutdown():
        start_time = time.time()

        try:
            image_data.frame_count += 1
            
            # Skip process_every_n_frames to reduce load
            if image_data.frame_count % image_data.process_every_n_frames != 0:
                rate.sleep()
                continue
                
            # extract line mask
            ll_seg_mask = model.detect( image_data.rgb_frame_cap)

            # publish drive lane line
            lane_line_img = bridge.cv2_to_imgmsg(ll_seg_mask, encoding="mono8")
            pub_line_mask.publish(lane_line_img)

            # # Display the image ----------------------------------------------------
            # cv2.imshow("Camera Image", image_data.rgb_frame_cap)
            # cv2.imshow('ll_seg_mask_normal', ll_seg_mask)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            processing_time = time.time() - start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0
            print(" Model process FPS : ", fps)

        except Exception as e:
            rospy.logerr(f"Processing error: {e}")
            traceback.print_exc()


        rate.sleep()
    
    cv2.destroyAllWindows()
    rospy.loginfo("Lane detection node shutting down")
    



if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

