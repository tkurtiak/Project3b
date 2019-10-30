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
from geometry_msgs.msg import Point
from std_msgs.msg import Empty

global_pos= Twist()#np.array([0.,0.,0.])
global_command= Twist()
global_vel=Twist()
global_waypoint=Point()

global_marker_center=np.array([0.,0.])
global_marker_center_avg=np.array([0.,0.])

islanded=True

pub_commands= rospy.Publisher('bebop/cmd_vel',Twist,queue_size=1)
pub_takeoff= rospy.Publisher('bebop/takeoff',Empty,queue_size=1)
pub_land= rospy.Publisher('bebop/land',Empty,queue_size=1) #trigger with pub_land.publish()
pub_waypoints= rospy.Publisher("/bebop/waypoint_ilya", Point, queue_size=1)

isfirststep=False

def quat_mult(a,b):
    
    c = np.array([0.,0.,0.,0.])
    c[0] = (a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3] )
    c[1] = (a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2] )
    c[2] = (a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1] )
    c[3] = (a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0] )
    return c

def callback(msg):
    global global_pos
    global global_vel
    rospy.loginfo(msg.pose.pose)
    rospy.loginfo(msg.twist.twist)

    #global_pos=np.array([msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.position.z])
    #print(global_pos)
    global_pos=msg.pose.pose
    global_vel=msg.twist.twist

def setlanded(msg):
    global islanded
    islanded=True
    pub_land.publish()

def updatecenter(msg):
    global global_marker_center
    global_marker_center[0]=msg.x
    global_marker_center[1]=msg.y
    rospy.loginfo(msg)


def odomlistener():
    #okay so we start this listener node
    global global_pos
    global islanded

    rospy.init_node('odomlistener_move', anonymous=True, log_level=rospy.WARN)
    rospy.Subscriber('/bebop/odom', Odometry, callback)
    rospy.Subscriber('/bebop/land',Empty, setlanded)
    rospy.Subscriber('/center_point',Point, updatecenter)
    print('node started')
    time.sleep(2.)
    print('taking off')
    pub_takeoff.publish()
    time.sleep(6.)
    islanded=False

    telemrate = 10
    rate = rospy.Rate(telemrate)
    # spin() simply keeps python from exiting until this node is stopped
    while not rospy.is_shutdown():
        # 4.3 and 2.1 from blue
        trackandland(1,0)

        rate.sleep()	


## Passing Initial position of the center of the circle from the drone (X,Y)
def trackandland(x,y):
    
    global isfirststep
    global global_pos
    global global_marker_center
    global global_marker_center_avg
    #x, y are initial coordinates of the landing (gaussian mean or whatever the fuck)

    FOVx=45 #deg
    FOVy=54


    while isfirststep!=True:
        print('Im doing the first steparoony now')
        # moveto_body(x,0,0) #move over ish the bitch
        # # -ve here is to the right
        # moveto_body(0,-y,0) #move over ish the bitch
        moveto_body(0,0,1.5) #move up to see it
        print('Im done with the first steparoony')
        isfirststep=True #first step is done, don't ever do this shit again

    while isfirststep==True:
        # if global_marker_center_avg[0] == 0 and global_marker_center_avg[1] == 0:
        #     global_marker_center_avg=global_marker_center #if its the first time than update the average flat
        # else:
        #     global_marker_center_avg=.8*global_marker_center_avg + .2*global_marker_center #keep a running average of the last 5 decent shits

        # print('In the testing loop!')
        
        #imgs are 320 x 240

        if global_marker_center[0] != 0 and global_marker_center[1] != 0:

            # print('In the testing loop!')
            # xcoord = global_marker_center[0]
            # ycoord = global_marker_center[1]

            # xdes = 320/2
            # ydes = 240/2
        
            # xvec = 0.05*(xdes - xcoord)
            # yvec = 0.05*(ydes - ycoord)

            # print("xvec is " + str(xvec))
            # print("yvec is " + str(yvec))

            # while not (xdes - 5 < xcoord < xdes + 5):
            #     moveto_body(xcoord-xvec,0,0)

            # while not (ydes - 5 < ycoord < ydes + 5):
            #     moveto_body(0,ycoord-yvec,0)

            vector2center= global_marker_center- np.array([160,120])
            # deg_offsetx= (vector2center[0]/160)*FOVx
            # deg_offsety= (vector2center[0]/120)*FOVy
            deg_offsets= (vector2center/(np.array([160,120])))*np.array([FOVx,FOVy])
            marker_loc=np.tan(deg_offsets*np.pi/180)*global_pos.position.z

            if np.linalg.norm(marker_loc)<.1:

                if global_pos.position.z>1.2:
                    pub_land.publish()
                    # moveto_body(0,0,-.5)
            else:
                moveto_body(.8*marker_loc[0],.8*marker_loc[1],0)

        # pub_land.publish()





def moveto_body(x,y,z):
    global global_pos
    global global_vel
    global global_command
    global islanded
    global global_waypoint


    #okay so have a vector in the actual body frame, want to convert to inertial/odom frame
    quat_B_to_I= np.array([global_pos.orientation.w, -global_pos.orientation.x, -global_pos.orientation.y, -global_pos.orientation.z])
    quat_B_to_I_inv= np.array([global_pos.orientation.w, global_pos.orientation.x, global_pos.orientation.y, global_pos.orientation.z])
    command_quat_body=np.array([0, x,y,z])
    temp= quat_mult(quat_B_to_I_inv,command_quat_body)
    command_quat_inertial= quat_mult(temp,quat_B_to_I)
    command_vect_inertial= command_quat_inertial[1:]

    #okay now have command in the odom frame, relative to body, so make it relative to origin

    expected_pos_inertial= np.array([global_pos.position.x, global_pos.position.y, global_pos.position.z]) + command_vect_inertial
    global_waypoint.x=expected_pos_inertial[0]
    global_waypoint.y=expected_pos_inertial[1]
    global_waypoint.z=expected_pos_inertial[2]

    #alright so this is the desired waypoint in the odometry frame

    move_array=np.array([0.,0.,0.])
    #print('STARTING MOVE')
    if islanded==False:
        print('STARTING MOVE')
        print(np.array([x,y,z]))

        error=1000.
        error_integral=np.array([0.,0.,0.])
        while error>.15 and islanded==False:# and np.linalg.norm(np.array([global_vel.linear.x, global_vel.linear.y, global_vel.linear.y]))>.1:
            print('error-------------------------------------------------------------------------')
            print(error)
            current_pos_inertial=np.array([global_pos.position.x, global_pos.position.y, global_pos.position.z])
            move_vect_inertial= expected_pos_inertial-current_pos_inertial

            #okay need to convert to vector in body frame to figure out where to move
            #using current orientation global_pos in case something is flukey
            quat_I_to_B= np.array([global_pos.orientation.w, global_pos.orientation.x, global_pos.orientation.y, global_pos.orientation.z])
            quat_I_to_B_inv= np.array([quat_I_to_B[0], -quat_I_to_B[1], -quat_I_to_B[2],-quat_I_to_B[3]])
            
            move_quat_inertial= np.array([0, move_vect_inertial[0],  move_vect_inertial[1], move_vect_inertial[2]])
            temp= quat_mult(quat_I_to_B_inv,move_quat_inertial)
            move_quat_body= quat_mult(temp,quat_I_to_B)
            move_vect_body= move_quat_body[1:]# this is basically your error vector


            #HOLD UP THE VELOCITY IS IN "CHILD FRAME" WHICH IS SOME OTHER BULLSHIT from HEADER FRAME
            # #we also have 
            # velocity_quat_inertial= np.array([0., global_vel.linear.x, global_vel.linear.y,global_vel.linear.z])
            # #so lets put it in body frame
            # temp2= quat_mult(quat_I_to_B_inv,velocity_quat_inertial)
            # velocity_quat_body= quat_mult(temp2,quat_I_to_B)
            # velocity_vect_body= velocity_quat_body[1:]# this is basically your error vector derivitive
            #velocity_vect_body= np.array([global_vel.linear.x, global_vel.linear.y,global_vel.linear.z])
            velocity_vect_body= np.array([global_vel.linear.x, global_vel.linear.y, global_vel.linear.z])
            error_integral=error_integral+move_vect_body


            #move_vect_body[2]=1.29*move_vect_body[2]

            move_array[0]=.08*move_vect_body[0] - .16*velocity_vect_body[0] + .001*error_integral[0] #TUNE THIS
            move_array[1]=.08*move_vect_body[1] - .16*velocity_vect_body[1] + .001*error_integral[1]
            move_array[2]=.53*move_vect_body[2] - .10*velocity_vect_body[2] + .001*error_integral[2]


            timedelay= .1#TUNE THIS
            print('move vect')
            print(move_array)
            print('move_vect_body')
            print(move_vect_body)
            print('velocity_vect_body')
            print(velocity_vect_body)
            print('error_integral')
            print(error_integral)
            print(' ')
            print('command is')
            print(np.array([x,y,z]))
            print('expected_pos_inertial')
            print(expected_pos_inertial)
            print('current_pos_inertial')
            print(current_pos_inertial)



            global_command.linear.x=move_array[0]
            global_command.linear.y=move_array[1]
            global_command.linear.z=move_array[2]
            global_command.angular.x=0
            global_command.angular.y=0
            global_command.angular.z=0

            if rospy.is_shutdown():
                break

            pub_commands.publish(global_command)
            time.sleep(timedelay)
            # pub_commands.publish(global_command)
            # time.sleep(timedelay)
            # pub_commands.publish(global_command)
            # time.sleep(timedelay)
            error= np.linalg.norm(np.array([global_pos.position.x, global_pos.position.y, global_pos.position.z])-expected_pos_inertial)
            #error=.4
            pub_waypoints.publish(global_waypoint)

        #print('TRYING TO LAND! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # pub_land.publish()
        # islanded=True
    else:
        print('Landed')
        pub_land.publish()
        islanded=True
    

if __name__ == '__main__':

    odomlistener()
    # rospy.spin()
