#!/usr/bin/env python3

# MPCC controller                    
# contour controlling

# 1. first consider only the following path
# 2. convert to the robot frame
# 3. apply MPC controller with contoue edit
# 4. find if there any collition in predicted path
# 5. if yes consider the respective states and update the weigth values of object
# 6. apply MPC controller with contour edit

# reference foxglove_QP_v4.0.0.py

import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import quaternion_from_euler, quaternion_multiply, euler_from_quaternion
import numpy as np
import QPC as QPC
import Frame_convert as Fc
import math
import time
import sys



np.set_printoptions(precision=3, suppress=True)



class MPC_controller:
    def __init__(self):
        self.robotGps_state = np.array([[0.0],[0.0],[0.0]])
        rospy.loginfo("Initializing state variables")
        # coordinate values should scale up by 10 to compare with Meter

        # pred horizon
        self.horizon    = 9
        
        # predicted control states (number) , [ [v], [w]]
        self.pred_control_val = np.tile( [[1],[0]], 16)

        # cost Fn control state constant
        self.control_val_R = np.zeros( len(self.pred_control_val[0])*2)

        # step time seconds
        self.dt = 0.025

        # cost fn state value constant (number)
        self.state_val_Q = np.array([   [  0.5,  0.5,    1,    1,  1.5,  1.5,    2,    2,  2.5,  2.5,    3,    3,  3.5,  3.5,    4,    4],
                                        [    1,    1,    2,    2,    3,    3,    4,    4,    5,    5,    6,    6,    7,    7,    8,    8],
                                        [ 0.05, 0.05, 0.10, 0.10, 0.15, 0.15, 0.20, 0.20, 0.25, 0.25, 0.30, 0.30, 0.35, 0.35, 0.40, 0.40] ])
        # diagonal matrix
        self.diag_state_val_Q = self.state_val_Q.flatten(order='F')
        self.diag_state_val_Q = np.diag( self.diag_state_val_Q)

        # updatable values ---------------------------------------------------------------
        # starting point
        self.X_0 = np.array([[0.0], [0.0], [0.0]])
        # self.X_0 = None

        # object coordinates
        # self.staticObj = np.array([ [  0.5],
        #                             [    0],
        #                             [    0] ])
        self.staticObj = None
        if self.staticObj is not None and self.X_0 is not None:
            # starting point to object distane
            for i in range( len( self.staticObj[0])):
                self.staticObj[2][i] = ( (self.staticObj[0][i] - self.X_0[0][0])**2 + (self.staticObj[1][i] - self.X_0[1][0])**2 )**(1/2)

        # reference state values [[ x],[ y],[ z]]
        self.sub_ref_path   = None
        self.ref_state_val  = None


        self.init_state_Robot   = None
        self.ref_state_val_Robot= None
        self.objStat_Robot      = None
        self.control_val        = None
        self.state_value        = None
        self.state_val_World    = None

    # update robot initial state
    def robocop_init(self):
        while self.robotGps_state[0][0] == 0:
            rospy.logwarn(" robocop gps values [(0.0),(0.0),(0.0)]")
        self.X_0    = self.robotGps_state

    # select the nearest ref point and get the suitable values
    def selected_refPath(self):
        distances = []
        for i in range(self.sub_ref_path.shape[1]):
            point = self.sub_ref_path[:2, i]
            distance = np.sqrt( (self.X_0[0][0] - point[0])**2 + (self.X_0[1][0] - point[1])**2  )  # Compare only x and y for distance
            distances.append(distance)

        # Find the index of the minimum distance
        min_index = np.argmin(distances)

        # Create a new ref_state_val array starting from the nearest point
        # self.ref_state_val = np.hstack(( self.sub_ref_path[:, min_index:], self.sub_ref_path[:, :min_index-1] ))
        self.ref_state_val = self.sub_ref_path[:, min_index:]

    # collistion test detect fn
    def collision_test(self, states, object_states):
        collision_list = []
        for object_num in range( len(object_states[0])):
            x_coord = object_states[0,object_num]
            y_coord = object_states[1,object_num]
            critical_step = np.empty((0,2))
            for state_num in range( len(states[0])):
                distance = ( (states[0,state_num] - x_coord)**2 + (states[1,state_num] - y_coord)**2)**(1/2)
                if ( distance < 0.1):
                    critical_step = np.vstack( (critical_step, np.array([state_num, distance]) ))
            collision_list.append(critical_step)

        return collision_list

    # update initial states, reference states and Q values by considering static objects
    def update_states(self, init_state, object_states, ref_states, val_Q, collision_list):
        # collision_list = self.collision_test( ref_states, object_states)
        dir_list = [-1, 1]

        for i in range( len(object_states[0])):
            if ( len(collision_list[i]) > 0):
                # update init_state
                init_state = np.vstack( (init_state, np.array([ [object_states[2][i]],[object_states[1][i]]]) ))

                # update state matrix
                states_update = np.zeros((2,17), dtype=np.float64)
                
                # update Q matrix
                val_Q_update = np.zeros((2,16))
                
                for num in collision_list[i]:
                    dir = dir_list[i]
                    states_update[0][ int(num[0])] = 0.1
                    states_update[1][ int(num[0])] = dir*(0.1 - num[1]/2)

                    val_Q[1][ int(num[0])-1] = 500
                    val_Q_update[0][ int(num[0])-1] = 100
                    val_Q_update[1][ int(num[0])-1] = 100

                ref_states = np.vstack(( ref_states, states_update))
                val_Q = np.vstack(( val_Q, val_Q_update))

        val_Q = val_Q.flatten(order='F')
        val_Q = np.diag( val_Q)

        return init_state, ref_states, val_Q

    # mpc solver
    def mpc_solver(self):
        if self.sub_ref_path is None:
            rospy.logwarn("referece state values are None")
            return
        elif self.X_0 is None:
            rospy.logwarn("initial state values are None")
            return
        else:
            self.selected_refPath()

        if len(self.ref_state_val[0]) <= 2:
            self.control_val    = np.array([[ 0.0, 0.0], [ 0.0, 0.0]])
            self.state_value    = np.array([[ 0.0, 0.0], [ 0.0, 0.0], [0.0, 0.0]])
            _, self.state_val_World = Fc.Convert_To_World_Frame( self.X_0.copy(), self.state_value.copy()[:3,:])
            rospy.logwarn("robocop at end point")

        else:
            try:
                # convert initial robot coord and reference coords to robot frame
                self.init_state_Robot, self.ref_state_val_Robot = Fc.Convert_To_Robot_Frame(    self.X_0.copy(), 
                                                                                                self.ref_state_val.copy())
                
                self.horizon = len( self.ref_state_val_Robot[0])

                # mpc solver
                rospy.loginfo("Solving the MPC controller")
                loop_count = 0
                while (True):
                    self.control_val, self.state_value = QPC.QPC_solutions( len(self.init_state_Robot.copy()),
                                                                            self.init_state_Robot.copy(),
                                                                            self.pred_control_val.copy()[:, :(self.horizon-1)],
                                                                            self.ref_state_val_Robot.copy()[:, :self.horizon],
                                                                            self.control_val_R.copy()[:(self.horizon-1)*2],
                                                                            self.diag_state_val_Q.copy()[:(self.horizon-1)*3, :(self.horizon-1)*3],
                                                                            self.dt  )
                    if ( np.isnan( self.control_val[0][0]) and loop_count < 10) :
                        loop_count = loop_count + 1
                        print( self.control_val[0][0], type( self.control_val[0][0]))
                    else :
                        break

                
                if self.staticObj is not None and self.staticObj.size !=0:
                    print( self.staticObj)
                    # convert initial object coord to robot frame
                    self.objStat_Robot = Fc.Convert_Obj_To_Robot_Frame( self.X_0.copy(), 
                                                                        self.staticObj.copy())
                    

                    criticals = self.collision_test(self.state_value.copy(), self.objStat_Robot.copy())
                    if any(critical_step.size > 0 for critical_step in criticals):  # Checks if any array has data
                        rospy.logwarn("There are collisions detected!")
                        
                        # update with static object properties
                        # init_state_Robot update - [x],[y],[theta],[l_k+1],[ly_k+1]
                        self.init_state_Robot_U, self.ref_state_val_Robot_U, self.diag_state_val_Q_U = self.update_states(  self.init_state_Robot.copy(), 
                                                                                                                            self.objStat_Robot.copy(), 
                                                                                                                            self.ref_state_val_Robot.copy()[:, :self.horizon], 
                                                                                                                            self.state_val_Q.copy(),
                                                                                                                            criticals )
                        
                        # mpc solver
                        rospy.loginfo("Solving the MPC controller")
                        loop_count = 0
                        while (True):
                            self.control_val, self.state_value = QPC.QPC_solutions( len(self.init_state_Robot_U.copy()),
                                                                                    self.init_state_Robot_U.copy(),
                                                                                    self.pred_control_val.copy(),
                                                                                    self.ref_state_val_Robot_U.copy(),
                                                                                    self.control_val_R.copy(),
                                                                                    self.diag_state_val_Q_U.copy(),
                                                                                    self.dt  )
                            if ( np.isnan( self.control_val[0][0]) and loop_count < 10) :
                                loop_count = loop_count + 1
                                print( self.control_val[0][0], type( self.control_val[0][0]))
                            else :
                                break

                    else:
                        rospy.logwarn("No collisions detected.")
                    
                    
                # convert reference coords and predicted coords to World frame
                _, self.state_val_World = Fc.Convert_To_World_Frame( self.X_0.copy(), self.state_value.copy()[:3,:])
                

            except Exception as e:
                rospy.logerr(f"mpc solver error : {e}") 

    # predict path points publisher
    def state_points(self):
        path_msg        = Float32MultiArray()
        points          = list(zip(self.state_val_World[0], self.state_val_World[1], self.state_val_World[2]))
        flattened_points= [coord for point in points for coord in point]
        path_msg.data   = flattened_points

        return path_msg

    # pred control value publisher
    def control_points(self):
        control_msg     = Float32MultiArray()
        points          = list(zip(self.control_val[0], self.control_val[1]))
        flattened_points= [coord for point in points for coord in point]
        control_msg.data= flattened_points

        return control_msg

    # predict path arrows publisher
    def get_markerArray(self):
        refMarkerArray = MarkerArray()
        refMarkerArray.markers = []

        for i in range( len(self.state_val_World[0])):
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
            marker.pose.position.x = self.state_val_World[0][i] * 10
            marker.pose.position.y = self.state_val_World[1][i] * 10
            marker.pose.position.z = 0

            # matrix shift
            q_orig = quaternion_from_euler(0, 0, -self.state_val_World[2][i])
            q_rot = quaternion_from_euler(math.pi, 0, 0)
            q_new = quaternion_multiply(q_rot, q_orig)
            # print(q_new)

            marker.pose.orientation.x = q_new[0]
            marker.pose.orientation.y = q_new[1]
            marker.pose.orientation.z = q_new[2]
            marker.pose.orientation.w = q_new[3]

            refMarkerArray.markers.append(marker)

        return refMarkerArray
        

# update the robot orientation
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

# update the robot potision
def pose_callback(msg):
        try:
            robot_pose = msg

            pose_x = robot_pose.pose.position.x
            pose_y = robot_pose.pose.position.y
            pose_z = robot_pose.pose.position.z

            mpc_viewer.X_0[0] = pose_x * 0.1
            mpc_viewer.X_0[1] = pose_y * 0.1
        except Exception as e:
            rospy.logerr(f"Error processing pose_callback: {e}")

# subscribe the reference path
def refPath_callback(msg):
        try:
            flattened_data = msg.data
            num_points = len(flattened_data) // 3
            if num_points > 17:
                mpc_viewer.horizon = 17
            else:
                mpc_viewer.horizon = num_points - 1

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

# robot gps local point
def robocop_gps_local_callback(msg):
    try:
        mpc_viewer.robotGps_state[0][0] = msg.data[0]/10
        mpc_viewer.robotGps_state[1][0] = msg.data[1]/10
    except Exception as e:
            rospy.logerr(f"Error processing robocop_gps_local_callback: {e}")

# subscribe the objects
def objects_camera_callback(msg):
    try:
        flattened_data = msg.data
        num_points = len(flattened_data) // 3

        # (x0,y0,0), (x1,y1,0), ....
        reshaped_data   = np.array(flattened_data).reshape(num_points, 3)

        obj_points      = [ reshaped_data[:, 0].tolist(),  # x coordinates
                            reshaped_data[:, 1].tolist(),  # y coordinates  
                            reshaped_data[:, 2].tolist()   # 0 value
                            ]
        
        mpc_viewer.staticObj    = np.array(obj_points)

        if mpc_viewer.staticObj is not None and mpc_viewer.X_0 is not None:
            # starting point to object distane
            for i in range( len( mpc_viewer.staticObj[0])):
                mpc_viewer.staticObj[2][i] = ( (mpc_viewer.staticObj[0][i] - mpc_viewer.X_0[0][0])**2 + (mpc_viewer.staticObj[1][i] - mpc_viewer.X_0[1][0])**2 )**(1/2)

    except Exception as e:
        rospy.logerr(f"Error processing refPath_callback: {e}")



def main():
    rospy.init_node('pred_path_node')

    rospy.loginfo("predicted path node started")
    rospy.loginfo("Published topics:")
    rospy.loginfo("  /robocop/pred_path/points - std_msgs.msg/Float32MultiArray")
    rospy.loginfo("  /robocop/pred_path/velocities - std_msgs.msg/Float32MultiArray")
    rospy.loginfo("  /robocop/pred_path/markers - visualization_msgs.msg/MarkerArray")
    

    robotImu_sub    = rospy.Subscriber('/zed2/zed_node/imu/data', Imu, callback = imu_callback)
    robotPose_sub   = rospy.Subscriber('/zed2/zed_node/pose', PoseStamped, callback = pose_callback)
    refPath_sub     = rospy.Subscriber('/robocop/ref_path/points', Float32MultiArray, callback = refPath_callback)
    roboLocal_sub   = rospy.Subscriber('/robocop/land_mark/robot', Float32MultiArray, callback = robocop_gps_local_callback)
    objCame_sub     = rospy.Subscriber('/robocop/object_det/points', Float32MultiArray, callback = objects_camera_callback)
    
    predPath_pub    = rospy.Publisher('/robocop/pred_path/points', Float32MultiArray, queue_size=10)
    predvelo_pub    = rospy.Publisher('/robocop/pred_path/velocities', Float32MultiArray, queue_size=10)
    predMark_pub    = rospy.Publisher('/robocop/pred_path/markers', MarkerArray, queue_size=10)

    rate = rospy.Rate(10)

    rospy.loginfo("Starting main processing loop")

    # mpc_viewer.robocop_init()
    
    while not rospy.is_shutdown():
        try:
            mpc_viewer.mpc_solver()

            predPath_pub.publish( mpc_viewer.state_points())
            predvelo_pub.publish( mpc_viewer.control_points())
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

