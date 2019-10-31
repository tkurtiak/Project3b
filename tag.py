#!/usr/bin/env python

import roslib; roslib.load_manifest('visualization_marker_tutorials')
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import rospy
import math

topic = 'visualization_marker_array'
publisher = rospy.Publisher(topic, Marker)

rospy.init_node('register')

markerArray = MarkerArray()

count = 0
MARKERS_MAX = 100

while not rospy.is_shutdown():

   marker = Marker()
   marker.header.frame_id = "/odom"
   # marker.type = marker.SPHERE
   # marker.action = marker.ADD
   marker.scale.x = 1
   marker.scale.y = 1
   marker.scale.z = 1
   marker.color.a = 0.0
   marker.color.r = 1.0
   marker.color.g = 1.0
   marker.color.b = 1.0
   marker.pose.orientation.w = 0.3
   marker.pose.position.x = 0.8
   marker.pose.position.y = 0.5
   marker.pose.position.z = 0.6

   marker.mesh_resource = "file://home/vdorbala/bebop_ws/src/ddmc/scripts/tagfinal.dae"
   marker.mesh_use_embedded_materials = True

   # We add the new marker to the MarkerArray, removing the oldest
   # marker from it when necessary
   # if(count > MARKERS_MAX):
   #     markerArray.markers.pop(0)

   # markerArray.markers.append(marker)

   # # Renumber the marker IDs
   # id = 0
   # for m in markerArray.markers:
   #     m.id = id
   #     id += 1

   # Publish the MarkerArray
   publisher.publish(marker)

   count += 1

   rospy.sleep(0.01)
