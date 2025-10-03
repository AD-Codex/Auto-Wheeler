#!/usr/bin/env python3



import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Imu
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np

bridge = CvBridge()

shape = (360, 640)
line_mask = np.empty(shape)



def laneLine_topic(msg):
    global line_mask

    rgb_img = msg
    line_mask = bridge.imgmsg_to_cv2(rgb_img, desired_encoding="mono8")


def separate_lane_masks(binary_mask):
    # Ensure the mask is in uint8 format (needed for connectedComponents)

    sahpe = binary_mask.shape
    filted_mask = np.zeros(sahpe)
    binary_mask = (binary_mask > 0).astype(np.uint8)  

    # Apply connected components to separate different lane lines
    num_labels, labels = cv2.connectedComponents(binary_mask)

    # Extract each lane as a separate mask
    masks = []
    for i in range(1, num_labels):  # Ignore background (label 0)
        single_mask = (labels == i).astype(np.uint8)
        masks.append(single_mask*i)

        filted_mask = filted_mask + single_mask*i

    return filted_mask, masks



if __name__ == '__main__':
    rospy.init_node('zed_node')
    rospy.loginfo("zed capture node start")

    sub_laneLine = rospy.Subscriber("/robocop/laneDetect_node/lane_line", Image, callback=laneLine_topic)

    rate = rospy.Rate(10)


    while not rospy.is_shutdown():

        filted_mask, masks = separate_lane_masks(line_mask)

        try:
            # Display the image
            cv2.imshow("Camera Image", (masks[0]*255).astype(np.uint8))
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        except Exception as e:
            rospy.logerr(f"Failed to display frame {e}")


    rospy.spin()