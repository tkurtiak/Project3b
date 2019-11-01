#!/usr/bin/env python
from __future__ import print_function

import sys
import rospy
# ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

# if ros_path in sys.path:

#     sys.path.remove(ros_path)

import cv2

# sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point

class image_converter:

    def __init__(self):
        rospy.init_node('image_converter1', anonymous=True)
        self.image_pub = rospy.Publisher("new_image",Image, queue_size=10)
        self.imageinput = None
        self.points = None
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/duo3d/left/image_rect",Image,self.callback)
        self.points_sub = rospy.Subscriber("/center_point",Point,self.callback2)

    def callback2(self,data):
        self.points = data

        self.compute()

    def callback(self,data):

        self.imageinput = data
        self.compute()

    def compute(self):

        if self.image_sub is not None and self.points_sub is not None and self.imageinput is not None and self.points is not None:
            try:
              cv_image = self.bridge.imgmsg_to_cv2(self.imageinput, "bgr8")
            except CvBridgeError as e:
              print(e)

            x = self.points.x
            y = self.points.y

            cv2.circle(cv_image,(int(x),int(y)),10,(0,0,255),-1)

            # cv2.imshow("new_image", cv_image)
            # k = cv2.waitKey()
            # if k == 27:
                # cv2.destroyAllWindows()

            try:
              self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
            except CvBridgeError as e:
              print(e)

def main(args):
  ic = image_converter()
  # rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)