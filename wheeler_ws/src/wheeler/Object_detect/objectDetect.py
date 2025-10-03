#!/usr/bin/env python3


import rospy
from zed_interfaces.msg import ObjectsStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point



def object_detect_node(msg):
    print("detecting...........")
    for obj in msg.objects:
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "zed_objects"
        marker.id = obj.label_id
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        corners = obj.bounding_box_3d.corners

        marker.points = []
        for i in range(8):
            p = Point()
            p.x = corners[i].kp[0]
            p.y = corners[i].kp[1]
            p.z = corners[i].kp[2]
            marker.points.append(p)
        

        pub_3d_box.publish(marker)



if __name__ == '__main__':
    rospy.init_node('objectDetect_node')
    rospy.loginfo("object detect node start")

    sub_object_data = rospy.Subscriber("/zed2i/zed_node/obj_det/objects", ObjectsStamped, callback=object_detect_node)
    pub_3d_box = rospy.Publisher("obj_detect/3d_box_marker", Marker, queue_size=10)

    rate = rospy.Rate(20)

    while not rospy.is_shutdown():
        rate.sleep()

    rospy.spin()