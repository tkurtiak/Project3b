#!/usr/bin/env python2
# vertex find
import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
from scipy.stats import multivariate_normal
from scipy.cluster.vq import kmeans, whiten, kmeans2
import imutils
#Image Select
import Tkinter, tkFileDialog
from time import time
import os
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError
import rospy

img_pub = rospy.Publisher("/center_point_img",Image)
center_pub = rospy.Publisher("/center_point",Point)

def findcenter(img):
    print('loaded image')
    #cv2.imshow('original image',img)

    
    #param2 is the threshold for circle detection
    whitedef=225
    avg_color = img.mean(axis=0).mean(axis=0)
    ##print('avg_color')
    #print(avg_color)
    # median = cv2.medianBlur(img,5)
    imgOG=img.copy()
    white_mask = cv2.inRange(img, np.array([whitedef,whitedef,whitedef]), np.array([255,255,255]))
    #cv2.imshow('white_mask',white_mask)


    white_mask = cv2.dilate(white_mask,np.ones((3,3), np.uint8),iterations=2)
    white_mask = cv2.erode(white_mask,np.ones((5,5), np.uint8),iterations=1)
    #cv2.imshow('white_mask after close',white_mask)

    x=0
    y=0
    cnts = cv2.findContours(255*white_mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    max_contour=None
    if len(cnts)>0:
        max_contour= max(cnts,key=cv2.contourArea)

    contour_img=0*imgOG.copy()
    if max_contour is not None:
        contour_img=cv2.drawContours(contour_img, [max_contour], -1, (255), -1)
        rect = cv2.minAreaRect(max_contour)

        sides=np.sort(rect[1])
        AR= sides[1]/sides[0]
        print(AR)
        
        if 1.18<AR<1.72:
            area = cv2.contourArea(max_contour)
            hull = cv2.convexHull(max_contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area)/hull_area
            print(solidity)
            x=int(rect[0][0])
            y=int(rect[0][1])
            if solidity>.8:
                cv2.circle(contour_img,(x,y),3,(255,0,255),-1)

    return contour_img, x, y

bridge = CvBridge()

def img_callback(data):
    img = bridge.imgmsg_to_cv2(data, "bgr8")
    img_center,x,y = findcenter(img)
    outt=Point()
    outt.x=x
    outt.y=y
    img_pub.publish(bridge.cv2_to_imgmsg(img_center, "bgr8"))
    center_pub.publish(outt)

def find_center():
    rospy.init_node('find_center', anonymous=True)
    img_sub = rospy.Subscriber("/duo3d/left/image_rect", Image, img_callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        find_center()
    except rospy.ROSInterruptException:
        pass
