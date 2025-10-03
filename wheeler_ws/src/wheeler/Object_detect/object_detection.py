#!/usr/bin/env python3


import rospy
from zed_interfaces.msg import ObjectsStamped
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np


class Person():
    def __init__(self):
        self.objects_data   = None

        self.staticObj      = None

    # extract the persons from object data
    def extrat_persons(self):
        if self.objects_data is None:
            rospy.logwarn("object data are None")
            return
        
        self.staticObj = np.empty((3, 0))

        for obj in self.objects_data.objects:
            if obj.label_id == 0:  # Person detected
                rospy.loginfo(f"Person detected with confidence: {obj.confidence}")

                corners = obj.bounding_box_3d.corners
                if len(corners) >= 8:
                    # Extract coordinates
                    x_coords = [corners[i].kp[0] for i in range(8)]
                    y_coords = [corners[i].kp[1] for i in range(8)]
                    z_coords = [corners[i].kp[2] for i in range(8)]
                        
                    # Calculate center
                    center_x = sum(x_coords) / 8  # Forward distance from camera
                    center_y = sum(y_coords) / 8  # Left along y axis
                    center_z = sum(z_coords) / 8  # Height

                    self.staticObj = np.hstack([ self.staticObj, [[center_x/10],[center_y/10],[0]]])

    # return object points
    def get_det_points(self):
        if self.staticObj is None:
            return

        det_msg        = Float32MultiArray()
        flattened_points= [coord for point in self.staticObj for coord in point]
        det_msg.data   = flattened_points

        return det_msg

    # return object markers
    def get_markerArray(self):
        if self.objects_data is None:
            return
        
        person_markerArray = MarkerArray()
        person_markerArray.markers = []

        self.marker_id_counter = 0
        
        for obj in self.objects_data.objects:
            if obj.label_id == 0:  # Person detected
                rospy.loginfo(f"Person detected with confidence: {obj.confidence}")

                cylinder_marker = Marker()
                cylinder_marker.header.frame_id = "world_link"
                cylinder_marker.header.stamp = rospy.Time.now()
                cylinder_marker.ns = "person_cylinders"
                cylinder_marker.id = self.marker_id_counter
                cylinder_marker.type = Marker.CYLINDER
                cylinder_marker.action = Marker.ADD

                corners = obj.bounding_box_3d.corners
                if len(corners) >= 8:
                    # Extract coordinates
                    x_coords = [corners[i].kp[0] for i in range(8)]
                    y_coords = [corners[i].kp[1] for i in range(8)]
                    z_coords = [corners[i].kp[2] for i in range(8)]
                        
                    # Calculate center
                    center_x = sum(x_coords) / 8  # Forward distance from camera
                    center_y = sum(y_coords) / 8  # Left along y axis
                    center_z = sum(z_coords) / 8  # Height
                        
                    # Set cylinder position
                    cylinder_marker.pose.position.x = center_x
                    cylinder_marker.pose.position.y = center_y
                    cylinder_marker.pose.position.z = center_z
                    cylinder_marker.pose.orientation.x = 0.0
                    cylinder_marker.pose.orientation.y = 0.0
                    cylinder_marker.pose.orientation.z = 0.0
                    cylinder_marker.pose.orientation.w = 1.0
                        
                    # Set cylinder dimensions
                    cylinder_diameter = 0.5
                    cylinder_marker.scale.x = 1
                    cylinder_marker.scale.y = 1
                    cylinder_marker.scale.z = 2
                        
                    # Set color (green for person)
                    cylinder_marker.color.r = 0.0
                    cylinder_marker.color.g = 1.0
                    cylinder_marker.color.b = 0.0
                    cylinder_marker.color.a = 0.7
                        
                    # Set lifetime
                    cylinder_marker.lifetime = rospy.Duration(1.0)
                        
                    person_markerArray.markers.append(cylinder_marker)
                    self.marker_id_counter += 1

        # Clear old markers if no objects detected
        if len(person_markerArray.markers) == 0:
            clear_marker = Marker()
            clear_marker.action = Marker.DELETEALL
            person_markerArray.markers.append(clear_marker)

        return person_markerArray





def objectDet_callback(msg):
    try:
        det_person.objects_data = msg
    except Exception as e:
        rospy.logerr(f"Error processing objectDet_callback: {e}")



def main():
    rospy.init_node('object_detection_node')

    rospy.loginfo("object detection node started")
    rospy.loginfo("Published topics:")
    rospy.loginfo("  /robocop/object_det/points - std_msgs.msg/Float32MultiArray")
    rospy.loginfo("  /robocop/object_det/markers - visualization_msgs.msg/MarkerArray")


    objDetect_sub   = rospy.Subscriber("/zed2/zed_node/obj_det/objects", ObjectsStamped, callback = objectDet_callback)
    
    objMarker_pub   = rospy.Publisher("/robocop/object_det/markers", MarkerArray, queue_size=10)
    objPoint_pub    = rospy.Publisher("/robocop/object_det/points", Float32MultiArray, queue_size=10)

    rate    = rospy.Rate(2)

    while not rospy.is_shutdown():
        try:
            det_person.extrat_persons()

            objPoint_pub.publish( det_person.get_det_points())
            objMarker_pub.publish( det_person.get_markerArray())
        except Exception as e:
            rospy.logerr(f"Processing error object_detection_node: {e}")
        
        rate.sleep()


if __name__ == '__main__':
    try:
        det_person = Person()
        main()
    except rospy.ROSInterruptException:
        pass

