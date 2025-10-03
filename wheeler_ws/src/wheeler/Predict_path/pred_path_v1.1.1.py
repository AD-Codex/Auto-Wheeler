#!/usr/bin/env python3

# MPCC controller                    
# contour controlling               -------- 1

# convert to robot frame
# apply MPCC controler
# convert to the world frame
# reference foxglove_QP_v1.1.1.py

import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import quaternion_from_euler, quaternion_multiply, euler_from_quaternion
import numpy as np
import QP_matrix as QPFn
import Frame_convert as Fc
import math


class MPC_controller:
    def __init__(self):
        rospy.loginfo("Initializing state variables")
        # coordinate values should scale up by 10 to compare with Meter

        self.horizon    = 15

        self.X_0 = np.array([ [0.4], [-0.1], [0.0]])
        self.dt = 0.05

        # reference state values [[ x],[ y],[ z]]
        self.sub_ref_path   = None
        self.ref_state_val  = None

        # U_predict [ [v], [w]]
        self.pred_control_val = np.array([  [ 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                                            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        # self.control_val_R = np.zeros( len(self.pred_control_val[0])*2)
        self.control_val_R = np.identity( len( self.pred_control_val[0])*2) *0.05

        self.state_val_Q = np.array([1,1,0.05, 5,5,0.05, 5,5,0.1, 10,10,0.1, 10,10,0.15, 15,15,0.15, 15,15,0.2, 20,20,0.2, 20,20,0.25, 25,25,0.25, 25,25,0.3, 30,30,0.3, 30,30,0.35, 35,35,0.35, 40,40,0.4])

        self.state_val_Q = np.diag( self.state_val_Q)

        self.o_state        = None
        self.RSV_Robot      = None
        self.state_value    = None
        self.control_val    = None
        self.SV_World       = None


    def selected_refPath(self):
        distances = []
        for i in range(self.sub_ref_path.shape[1]):
            point = self.sub_ref_path[:2, i]
            distance = np.sqrt( (self.X_0[0][0] - point[0])**2 + (self.X_0[1][0] - point[1])**2  )  # Compare only x and y for distance
            distances.append(distance)

        # Find the index of the minimum distance
        min_index = np.argmin(distances)

        # Create a new ref_state_val array starting from the nearest point
        self.ref_state_val = np.hstack(( self.sub_ref_path[:, min_index:], self.sub_ref_path[:, :min_index-1] ))


    def mpc_solver(self):
        if self.sub_ref_path is None:
            rospy.logwarn("referece state values are None")
            return
        else:
            self.selected_refPath()

        try:
            # convert initial robot coord and reference coords to robot frame
            self.o_state, self.RSV_Robot = Fc.Convert_To_Robot_Frame( self.X_0.copy(), self.ref_state_val.copy())

            # mpc solver
            rospy.loginfo("Solving the MPC controller")
            while (True):
                self.control_val, self.state_value = QPFn.QPC_solutions( self.o_state.copy(), self.dt, self.pred_control_val.copy(), self.RSV_Robot.copy()[:,:16], self.control_val_R.copy(), self.state_val_Q.copy())
                if ( np.isnan( self.control_val[0][0])) :
                    print( self.control_val[0][0], type( self.control_val[0][0]))
                else :
                    break

            # convert reference coords and predicted coords to World frame
            _, self.SV_World = Fc.Convert_To_World_Frame( self.X_0.copy(), self.state_value.copy())

        except Exception as e:
            rospy.logerr(f"mpc solver error : {e}") 


    def get_points(self):
        path_msg        = Float32MultiArray()
        points          = list(zip(self.SV_World[0], self.SV_World[1], self.SV_World[2]))
        flattened_points= [coord for point in points for coord in point]
        path_msg.data   = flattened_points

        return path_msg


    def get_markerArray(self):
        refMarkerArray = MarkerArray()
        refMarkerArray.markers = []

        for i in range( len(self.SV_World[0])):
            marker = Marker()
            marker.header.frame_id  = 'world_link'
            marker.header.stamp = rospy.Time.now()
            marker.ns = ""

            # Shape
            marker.type = Marker.ARROW
            marker.id = i
            marker.action = Marker.ADD

            # Scale
            marker.scale.x = 0.2
            marker.scale.y = 0.02
            marker.scale.z = 0.02

            # Color
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            # Pose
            marker.pose.position.x = self.SV_World[0][i] * 10
            marker.pose.position.y = self.SV_World[1][i] * 10
            marker.pose.position.z = 0

            # matrix shift
            q_orig = quaternion_from_euler(0, 0, -self.SV_World[2][i])
            q_rot = quaternion_from_euler(math.pi, 0, 0)
            q_new = quaternion_multiply(q_rot, q_orig)
            # print(q_new)

            marker.pose.orientation.x = q_new[0]
            marker.pose.orientation.y = q_new[1]
            marker.pose.orientation.z = q_new[2]
            marker.pose.orientation.w = q_new[3]

            refMarkerArray.markers.append(marker)

        return refMarkerArray
        


def imu_callback(msg):
        try:
            robot_imu = msg

            qx = robot_imu.orientation.x
            qy = robot_imu.orientation.y
            qz = robot_imu.orientation.z
            qw = robot_imu.orientation.w

            quaternion  = (qx, qy, qz, qw)

            roll, yaw, pitch = euler_from_quaternion(quaternion)
            mpc_viewer.X_0[2] = pitch
        except Exception as e:
            rospy.logerr(f"Error processing imu_callback: {e}")

def pose_callback(msg):
        try:
            robot_pose = msg

            pose_x = robot_pose.pose.position.x
            pose_y = robot_pose.pose.position.y
            pose_z = robot_pose.pose.position.z

            mpc_viewer.X_0[0] = pose_x
            mpc_viewer.X_0[1] = pose_y
        except Exception as e:
            rospy.logerr(f"Error processing pose_callback: {e}")

def refPath_callback(msg):
        try:
            flattened_data = msg.data
            num_points = len(flattened_data) // 3

            # (x0,y0,z0), (x1,y1,z1), ....
            reshaped_data = np.array(flattened_data).reshape(num_points, 3)

            # list set
            path_points = [ reshaped_data[:, 0].tolist(),  # x coordinates
                            reshaped_data[:, 1].tolist(),  # y coordinates  
                            reshaped_data[:, 2].tolist()   # theta coordinates
                            ]
            
            mpc_viewer.sub_ref_path  = np.array(path_points)

        except Exception as e:
            rospy.logerr(f"Error processing refPath_callback: {e}")



def main():
    rospy.init_node('pred_path_node')

    rospy.loginfo("predicted path node started")
    rospy.loginfo("Published topics:")

    robotImu_sub    = rospy.Subscriber('/zed2/zed_node/imu/data', Imu, callback = imu_callback)
    robotPose_sub   = rospy.Subscriber('/zed2/zed_node/pose', PoseStamped, callback = pose_callback)
    refPath_sub     = rospy.Subscriber('/robocop/ref_path/points', Float32MultiArray, callback= refPath_callback)
    
    predPath_pub    = rospy.Publisher('/robocop/pred_path/points', Float32MultiArray, queue_size=10)
    predvelo_pub    = rospy.Publisher('/robocop/pred_path/velocities', Float32MultiArray, queue_size=10)
    predMark_pub    = rospy.Publisher('/robocop/pred_path/markers', MarkerArray, queue_size=10)

    rate = rospy.Rate(1)

    rospy.loginfo("Starting main processing loop")
    
    while not rospy.is_shutdown():
        try:
            mpc_viewer.mpc_solver()

            predPath_pub.publish( mpc_viewer.get_points())
            predMark_pub.publish( mpc_viewer.get_markerArray())
        except Exception as e:
            rospy.logerr(f"Processing error: {e}")
        
        rate.sleep()


if __name__ == '__main__':
    try:
        mpc_viewer = MPC_controller()
        main()
    except rospy.ROSInterruptException:
        pass

