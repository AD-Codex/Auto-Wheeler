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
    binary_mask = binary_mask.astype(np.uint8)
    _, binary_mask = cv2.threshold( binary_mask, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    separated_lines = []
    for i, contour in enumerate(contours):
        # filter noise
        if cv2.contourArea(contour) > 500:
            line_image = np.zeros_like(binary_mask)
            cv2.drawContours( line_image, [contour], -1, 255, thickness=cv2.FILLED)

            separated_lines.append(line_image)

    return separated_lines



if __name__ == '__main__':
    rospy.init_node('zed_node')
    rospy.loginfo("zed capture node start")

    sub_laneLine = rospy.Subscriber("/robocop/laneDetect_node/lane_line", Image, callback=laneLine_topic)

    rate = rospy.Rate(10)


    while not rospy.is_shutdown():

        separated_lines = separate_lane_masks(line_mask)

        try:
            for i, line_img in enumerate(separated_lines):
                cv2.imshow("Line "+str(i), line_img)
            # # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        except Exception as e:
            rospy.logerr(f"Failed to display frame {e}")


    rospy.spin()