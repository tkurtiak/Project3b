#!/usr/bin/env python


## Helical Trajectory Commander that listens to Bebop odometry messages
## And commands bebop pitch/roll/yaw to achieve a Helix

# Make this program exacutable
# chmod +x nodes/HelixTrajectoryController.py

import rospy
import time
import numpy as np
#import tf  # use tf to easily calculate orientations


from nav_msgs.msg import Odometry # We need this message type to read position and attitude from Bebop nav_msgs/Odometry
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from std_msgs.msg import Empty

global_pos= Pose()#np.array([0.,0.,0.])
global_targetpose= Twist()

global_marker_center=np.array([0.,0.])

pub_pose= rospy.Publisher('/target_pose',Twist,queue_size=1)


def callback(msg):
    global global_pos
    global global_vel

    #global_pos=np.array([msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.position.z])
    #print(global_pos)
    global_pos=msg.pose.pose
    global_vel=msg.twist.twist


def updatecenter(msg):
    global global_marker_center
    global_marker_center[0]=msg.x
    global_marker_center[1]=msg.y


def odomlistener():
    #okay so we start this listener node
    


    rospy.init_node('target_pose', anonymous=True, log_level=rospy.WARN)
    rospy.Subscriber('/bebop/odom', Odometry, callback)
    rospy.Subscriber('/center_point',Point, updatecenter)
    print('node started')
    
    telemrate = 10
    rate = rospy.Rate(telemrate)
    # spin() simply keeps python from exiting until this node is stopped
    while not rospy.is_shutdown():
        
        dothething()

        rate.sleep()	

def dothething():
    global global_pos
    global global_marker_center
    global global_targetpose
    FOVx=45 #deg
    FOVy=54

    print(global_pos)
    print(global_marker_center)


    vector2center= np.array([(120-global_marker_center[1]),(160-global_marker_center[0])])
    deg_offsets= (vector2center/(np.array([160,120])))*np.array([FOVx,FOVy])
    marker_loc=np.tan(deg_offsets*np.pi/180)*global_pos.position.z




    global_targetpose.linear.x=marker_loc[0]
    global_targetpose.linear.y=marker_loc[1]
    global_targetpose.linear.z=-global_pos.position.z
    global_targetpose.angular.x=0
    global_targetpose.angular.y=0
    global_targetpose.angular.z=0

    pub_pose.publish(global_targetpose)

if __name__ == '__main__':

    odomlistener()
    # rospy.spin()
