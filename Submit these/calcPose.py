
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

a = .5
b = .500002
x0 = 0
y0 = 2

A = b**2
B = 0
C = a**2
D = 2*A*x0-B*y0
E = -B*x0-2*C*y0
F = A*x0**2+B*x0*y0+C*y0**2-a**2*b**2
C = [[A, .5*B, .5*D], 
    [.5*B, C, .5*E],
    [.5*D,.5*E,F]]



r = 1
# C = [[1, 0, 0], 
#     [0, 1.001, 0],
#     [0,0,1]]


# def calcPose(C,r):
# 	# C is symmetric ellipse matrix
# 	# K is camera calibration matrix
# 	# r = known radius



# 	U, S, V = np.linalg.svd(C)
	
# 	g = np.sqrt((S[1]-S[2])/(S[0]-S[2]))
# 	h = np.sqrt((S[0]-S[1])/(S[0]-S[2]))
# 	i = np.sqrt((S[0]-S[1])*(S[1]-S[2])/(S[0]*S[2]))#(-S[0]*S[2]))

# 	# discrete signs +1 or -1.  These are the 8 possible solutions.  
# 	s1 = +1
# 	s2 = +1
# 	s3 = +1

# 	R = V*np.array(([[g,s1*g,s2*h],[0,-s1,0],[s1*s2*h,s2*h,s1*g]]))
# 	t = [-s2*s3*i*r,-s1*s2*s3*i*r,s3*r*S[1]/np.sqrt(S[0]*S[2])]#np.sqrt(-S[0]*S[2])]


# 	return R,t


# R,t = calcPose(C,r)
# print(R)
# print(t)


x = np.array([[1,2,3,4],[2,3,4,5],[1,1,1,1]])
X = np.array([[7,4,7,8],[1,8,6,-12],[0,0,0,0]])
K = np.array([[1, 0, 0],
       [0, 1, 0],
       [0, 0, 1]])

# def linearPnP(K,x,X):
# 	u = x[0]
# 	v = x[1]
# 	Xt = np.transpose(X)
# 	[0,0,0,0,-Xt,np.dot(u,Xt)]


# 	# Pad the 3d Points X with a trailing row of ones
# 	pt3D = np.ones([4,4])
# 	pt3D[0:3,:] = X
# 	# Use linear algebra solver to solve for R|t matrix
# 	Rt = np.linalg.solve(K,np.dot(x,np.linalg.inv(pt3D)))
# 	R = Rt[:3,:3]
# 	U, s, V = np.linalg.svd(R)
# 	Rcorrected = np.dot(U,np.transpose(V))
# 	t = Rt[:3,3]

# 	return R,t

def Homography(K,x,X):
# takes in n number of x,y coords from image plane (x) and world plane (X).
	# initialize A matrix
	A = np.zeros((2*x.shape[1],9))
	for i in range(0,x.shape[1]-1):
		# undistort image plane points with K matrix
		x[:,i] = np.dot(np.linalg.inv(K),x[:,i])
		A[2*i,:] = np.array([-X[0,i],-X[1,i],-1, 0,0,0,X[0,i]*x[0,i],X[1,i]*x[0,i],x[0,i]])
		A[2*i+1,:] = np.array([0,0,0,-X[0,i],-X[1,i],-1,X[0,i]*x[1,i],X[1,i]*x[1,i],x[1,i]])
	U, s, V = np.linalg.svd(A)

	Hvec = V[:,8]
	# Reshape H into matrix 3x3
	H = np.reshape(Hvec,(3,3))

	return H

# if t is known, pose can be calculated as such:
#R = H-t*n # where n is [0,0,1] normal to the bullseye

def linearPnP(K,x,X):
# takes in n number of x,y coords from image plane (x) and world plane (X).
# based on a formula from https://cmsc733.github.io/assets/2019/p3/results/pdf/khoi_sgteja_p3.pdf

	# initialize A matrix
	A = np.zeros((3*x.shape[1],12))
	for i in range(0,x.shape[1]-1):
		# undistort image plane points with K matrix
		x[:,i] = np.dot(np.linalg.inv(K),x[:,i])
		A[2*i,:] = np.array([0,0,0,0,-X[0,i],-X[1,i],-X[2,i],-1,x[1,i]*X[0,i],x[1,i]*X[1,i],x[1,i]*X[2,i],x[1,i]])
		A[2*i+1,:] = np.array([X[0,i],X[1,i],X[2,i],1,0,0,0,0,-x[0,i]*X[0,i],-x[0,i]*X[1,i],-x[0,i]*X[2,i],-x[0,i]])
		A[2*i+1,:] = np.array([-x[1,i]*X[0,i],-x[1,i]*X[1,i],-x[1,i]*X[2,i],-x[1,i],x[0,i]*X[0,i],x[0,i]*X[1,i],x[0,i]*X[2,i],x[0,i],0,0,0,0])
	U, s, V = np.linalg.svd(A)

	Temp = V[:,11]
	#print(Temp)
	RT = np.reshape(Temp,(3,4))
	print(RT)
	R = RT[:,0:3]
	T = np.array([RT[:,3]])
	T = T.T

	Ru, Rs, Rv = np.linalg.svd(R)
	Rorthog = np.dot(Ru,np.transpose(Rv))
	if np.linalg.det(Rorthog) == -1:
		Rorthog = -Rorthog
		T = -T
	C = np.dot(np.transpose(-Rorthog),T) # center of camera position

	return Rorthog, C, T

#cv2.solvePnP(X[:3,:],x[:3,:],K,null)
# # rvec and tvec will give you the position of the world frame(defined at the center of the window) relative to the camera frame 
dist_coeff = np.array([0,0,0,0])
_res, rvec, tvec = cv2.solvePnP(np.transpose(np.float32(X)), np.transpose(np.float32(x[:2])), np.float32(K), np.float32(dist_coeff), None, None, False, cv2.SOLVEPNP_ITERATIVE)
R_pnp = cv2.Rodrigues(rvec)
Qpnp = np.concatenate((R_pnp[0],tvec),axis = 1)
X1 = np.append(X,[[1,1,1,1]],axis = 0)

upnp = np.dot(np.dot(K,Qpnp),X1)

Rorthog, C, T = linearPnP(K,x,X)
Q = np.concatenate([Rorthog,T],axis = 1)
uTKpnp = np.dot(np.dot(K,Q),X1)

from geometry_msgs.msg import Pose

def calcBodyPose(Rin,Tin):
	pose = Pose()
	cRb = np.transpose(np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]))
	cRi = Rin
	bRi = np.transpose(cRb)*Rin
	bti = np.dot(cRb,Tin)

	iRb = np.transpose(bRi)
	itb = -np.transpose(bRi)*bti
	print(iRb)
	rvec_fin = cv2.Rodrigues(iRb)
	tvec_fin = itb

	pose.position.x = tvec_fin[0]
	pose.position.y = tvec_fin[1]
	pose.position.z = tvec_fin[2]
	print(rvec_fin)
	print(np.reshape(rvec_fin[0],3))
	rquat = R.from_rotvec(np.reshape(rvec_fin[0],3))
	rquat = rquat.as_quat()
	# #print(rquat)

	pose.orientation.w = rquat[3]
	pose.orientation.x = rquat[0]
	pose.orientation.y = rquat[1]
	pose.orientation.z = rquat[2]
	return pose

posePnp = calcBodyPose(R_pnp[0],tvec)
poseTK = calcBodyPose(Rorthog,T)