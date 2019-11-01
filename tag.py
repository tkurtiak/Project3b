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
    marker.type = marker.MESH_RESOURCE
    marker.mesh_resource = "package://ddmc/meshes/tag.dae"
    marker.mesh_use_embedded_materials = True
    # marker.action = marker.ADD
    marker.scale.x = 1
    marker.scale.y = 1
    marker.scale.z = 1
    # marker.color.a = 1.0
    # marker.color.r = 1.0
    # marker.color.g = 0.0
    # marker.color.b = 0.0
    marker.pose.orientation.w = 1
    marker.pose.position.x = 1.5
    marker.pose.position.y = -3.5
    marker.pose.position.z = 0.0

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
    rospy.sleep(0.01)
