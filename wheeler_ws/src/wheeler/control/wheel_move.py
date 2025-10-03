#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from diagnostic_msgs.msg import DiagnosticArray
from std_msgs.msg import Float32MultiArray
import numpy as np



class Robot:
    def __init__(self):
        self.joy_state = 0

        self.linear_break = 0
        self.linear_x = 0
        self.pre_linear_x = 0
        self.angular_z = 0
        self.pre_angular_z = 0

        self.mpc_mode   = 0
        self.pred_state = 0
        self.pred_velo  = 0



# joy availablity check
def joy_available(msg : DiagnosticArray):
    global joy_state

    print(msg.status[0].level)
    eWheeler.joy_state = msg.status[0].level

    if (eWheeler.joy_state == 0):
        print("joy available")
    else :
        print("joy unavailable")

# joystick callback fn
def joy_callback(msg : Joy):
    global linear_x
    global angular_z
    global linear_break

    eWheeler.linear_x = msg.axes[1]
    eWheeler.angular_z = msg.axes[3]

    eWheeler.mpc_mode

    if ( msg.buttons[4] == 1 or msg.buttons[5] == 1) :
        eWheeler.mpc_mode = 1
    else :
        eWheeler.mpc_mode = 0

    if ( msg.buttons[6] == 1 or msg.buttons[7] == 1) :
        eWheeler.linear_break = 1
    else :
        eWheeler.linear_break = 0

    # print("break:", eWheeler.linear_break ," W_linear vel:", eWheeler.linear_x, " S_angular vel:", eWheeler.angular_z)

# arduino feed back
def feedback(msg):
    rospy.loginfo("Received message: %s", msg.data)

# pred path call back
def predPath_callback(msg):
    try:
        flattened_data = msg.data
        num_points = len(flattened_data) // 3

        # (x0,y0,z0), (x1,y1,z1), ....
        reshaped_data = np.array(flattened_data).reshape(num_points, 3)
        eWheeler.pred_state  = reshaped_data[1]
    except Exception as e:
        rospy.logerr(f"Error processing predPath_callback: {e}")

# ored velocity call back
def predVelo_callback(msg):
    try:
        flattened_data = msg.data
        num_points = len(flattened_data) // 2

        # (v0,w0), (v1,w1), ....
        reshaped_data = np.array(flattened_data).reshape(num_points, 2)
        eWheeler.pred_velo  = reshaped_data[0]
    except Exception as e:
            rospy.logerr(f"Error processing predVelo_callback: {e}")


def robot_control():
    wheeler_cmd = Twist()

    try:
        alpha_x = 0.1
        alpha_z = 0.1

        error_x = eWheeler.linear_x - eWheeler.pre_linear_x
        error_z = eWheeler.angular_z - eWheeler.pre_angular_z

        cur_linear_x = round( eWheeler.pre_linear_x + error_x*alpha_x, 3)
        cur_angular_z = round( eWheeler.pre_angular_z + error_z*alpha_z, 3)

        eWheeler.pre_linear_x = cur_linear_x
        eWheeler.pre_angular_z = cur_angular_z

        # print("break:", linear_break ," W_linear vel:", cur_linear_x, " S_angular vel:", cur_angular_z)

        if ( eWheeler.linear_break == 0 and eWheeler.joy_state == 0) :
            wheeler_cmd.linear.x = cur_linear_x * 50
            wheeler_cmd.angular.z = cur_angular_z * 100
        else :
            wheeler_cmd.linear.x = 0
            wheeler_cmd.angular.z = 0
            eWheeler.linear_x = 0
            eWheeler.angular_z = 0
            eWheeler.pre_linear_x = 0
            eWheeler.pre_angular_z = 0
    except Exception as e:
        rospy.logerr(f"Error robot_control controlling: {e}")

    return wheeler_cmd


def robot_auto():
    wheeler_cmd = Twist()

    try:
        auto_linearX  = eWheeler.pred_velo[0] / 10
        auto_anglarZ  = eWheeler.pred_state[2] / 0.532

        alpha_x = 0.1
        alpha_z = 0.1

        error_x = auto_linearX - eWheeler.pre_linear_x
        error_z = auto_anglarZ - eWheeler.pre_angular_z

        cur_linear_x = round( eWheeler.pre_linear_x + error_x*alpha_x, 3)
        cur_angular_z = round( eWheeler.pre_angular_z + error_z*alpha_z, 3)

        eWheeler.pre_linear_x = cur_linear_x
        eWheeler.pre_angular_z = cur_angular_z

        # print("break:", linear_break ," W_linear vel:", cur_linear_x, " S_angular vel:", cur_angular_z)

        if ( eWheeler.linear_break == 0 and eWheeler.joy_state == 0) :
            wheeler_cmd.linear.x    = cur_linear_x * 100
            wheeler_cmd.angular.z   = cur_angular_z * 100
        else :
            wheeler_cmd.linear.x = 0
            wheeler_cmd.angular.z = 0
            eWheeler.linear_x = 0
            eWheeler.angular_z = 0
            eWheeler.pre_linear_x = 0
            eWheeler.pre_angular_z = 0
    except Exception as e:
        rospy.logerr(f"Error robot_auto controlling: {e}")

    return wheeler_cmd


        

if __name__ == '__main__':
    rospy.init_node('wheel_move_node')
    rospy.loginfo("wheel move start")

    Joy_sub = rospy.Subscriber("/joy", Joy, callback=joy_callback)
    Joy_available = rospy.Subscriber("/diagnostics", DiagnosticArray, callback=joy_available)
    ard_sub = rospy.Subscriber("/ewheeler/feedback", String, callback=feedback)

    mpcPath_sub= rospy.Subscriber("/wheeler/pred_path/points", Float32MultiArray, callback=predPath_callback)
    mpcVelo_sub = rospy.Subscriber("/wheeler/pred_path/velocities", Float32MultiArray, callback=predVelo_callback)

    cmd_pub = rospy.Publisher("/ewheeler/cmd_vel", Twist, queue_size=10)

    eWheeler = Robot()

    rate = rospy.Rate(20)

    while not rospy.is_shutdown():
        if ( eWheeler.mpc_mode == 1):
            cmd_pub.publish( robot_auto())
        else:
            cmd_pub.publish( robot_control())

        rate.sleep()

    rospy.spin()
