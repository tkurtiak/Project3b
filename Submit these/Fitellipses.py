#!/usr/bin/env python2
# vertex find
import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
from scipy.stats import multivariate_normal
from scipy.linalg import eig
#from scipy.cluster.vq import kmeans, whiten, kmeans2
import imutils
#Image Select
import Tkinter, tkFileDialog
from time import time
import os



C_matrix=np.zeros((6,6))
C_matrix[2,0]=2
C_matrix[1,1]=-1
C_matrix[0,2]=2
#C_matrix_inv= np.linalg.inv(C_matrix)

def findcircles(img):
	#print('loaded image')
	#cv2.imshow('original image',img)
	whitedef=220
	avg_color = img.mean(axis=0).mean(axis=0)
	##print('avg_color')
	#print(avg_color)
	# median = cv2.medianBlur(img,5)
	imgOG=img.copy()

	#mask out white stuff
	white_mask = cv2.inRange(img, np.array([whitedef,whitedef,whitedef]), np.array([255,255,255]))
	#cv2.imshow('white_mask',white_mask)

	white_maskOG=white_mask.copy()
	#close oepration because edges near marker get a little ghosty
	white_mask = cv2.dilate(white_mask,np.ones((3,3), np.uint8),iterations=2)
	white_mask = cv2.erode(white_mask,np.ones((5,5), np.uint8),iterations=1)
	#cv2.imshow('white_mask after close',white_mask)

	#find the biggest external contour (presumably the target)
	cnts = cv2.findContours(255*white_mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	max_contour=None
	if len(cnts)>0:
		max_contour= max(cnts,key=cv2.contourArea)
	contour_img=0*white_mask.copy()
	if max_contour is not None:
		contour_img=cv2.drawContours(contour_img, [max_contour], -1, (1), -1)
	#check if it's a good match
	#using, AR, solidity, and total area
	rect = cv2.minAreaRect(max_contour)
	sides=np.sort(rect[1])
	AR= sides[1]/sides[0]
	print(AR)

	# c1,c2 =splitContourLeftRight(max_contour)


	
	marker_center=None
	if 1.18<AR<1.72:
		area = cv2.contourArea(max_contour)
		hull = cv2.convexHull(max_contour)
		hull_area = cv2.contourArea(hull)
		solidity = float(area)/hull_area
		print(solidity)
		print(area)
		if solidity>.8 and area>40:
			marker_center=np.array([int(rect[0][0]),int(rect[0][1])])
			base_marker_mask=cv2.bitwise_and(white_maskOG, white_maskOG, mask = contour_img)
			cv2.imshow('base_marker_mask',base_marker_mask)

	#if its okay then we in buisness to locate some circles. 
	edges_blank=0*img.copy() #draw contours on this canvas 
	output=edges_blank.copy()
	if marker_center is not None:

		print('marker_center')
		print(marker_center)
		cnts = cv2.findContours(255*base_marker_mask.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		
		equi_radius= np.sqrt(area/np.pi)
		


		# cv2.drawContours(edges_blank, [c1], -1, (120,0,255), 3)
		# cv2.drawContours(edges_blank, [c2], -1, (120,255,0), 3)


		for c in cnts:
			#cv2.drawContours(edges_blank, [c], -1, (255,0,0), 1)	#blue contours are filtered out		
			M = cv2.moments(c)
			cx = int(M['m10']/(M['m00']+.000001))
			cy = int(M['m01']/(M['m00']+.000001))
			dist=np.linalg.norm(np.array([cx-marker_center[0],cy-marker_center[1]]))
			if dist < equi_radius:
				center,axes,theta,error=fitEllipse(c)

				if error>.03:
					#cv2.ellipse(output,center,axes,theta,0,360,(0,255,0),1) #green ones are shitty ellipses
					# if :
					# 	cv2.drawContours(edges_blank, [c], -1, (0,255,0), 1)

					if error>999 or np.abs(cx-marker_center[0])<1:
						cv2.drawContours(edges_blank, [c], -1, (130,255,12), 2)
					if M['m00']< .9*area:
						cv2.drawContours(edges_blank, [c], -1, (0,255,0), 1)
						if ((cx+cy)>(marker_center[1]+marker_center[0]) and (cy-cx) < (marker_center[1]-marker_center[0]))  or  ((cx+cy)<(marker_center[1]+marker_center[0]) and (cy-cx) > (marker_center[1]-marker_center[0])):
							#side by side split
							c1,c2=splitContourLeftRight(c)
							print('leftright')
							print(cx,cy)
						else:
							#top down
							c1,c2=splitContourUpDown(c)
							print('updown')
							print(cx,cy)

						print('splitty boi')
						center1,axes1,theta1,error1=fitEllipse(c1)
						center2,axes2,theta2,error2=fitEllipse(c2)

						if error1<error2:
							cv2.drawContours(edges_blank, [c1], -1, (20,155,15), 1)
							if error1<.085 and np.linalg.norm(np.array([center1[0]-marker_center[0],center1[1]-marker_center[1]])) < .2*equi_radius:
								cv2.ellipse(output,center1,axes1,theta1,0,360,(95,55,15),1) #these ones are halfy boi ellipses
								cv2.drawContours(edges_blank, [c1], -1, (95,55,15), 1)
								
						else:
							cv2.drawContours(edges_blank, [c2], -1, (20,155,15), 1)
							if error2<.085 and np.linalg.norm(np.array([center2[0]-marker_center[0],center2[1]-marker_center[1]])) < .2*equi_radius:
								cv2.ellipse(output,center2,axes2,theta2,0,360,(95,55,15),1) #these ones are halfy boi ellipses
								cv2.drawContours(edges_blank, [c2], -1, (95,55,15), 1)
								

				else:

					if dist<.2*equi_radius:
						cv2.ellipse(output,center,axes,theta,0,360,(255,0,255),1) #pink ones are danky panky ellipses
						cv2.drawContours(edges_blank, [c], -1, (255,0,255), 1)
				

				#cv2.drawContours(output, [c], -1, (255,255,255), 1)
				
				#cv2.ellipse(output,center,axes,theta,0,360,(255,0,255),1)
	
	cv2.imshow('contours from edges',edges_blank)
	cv2.imshow('output',output)




def fitEllipse(contour):
	global C_matrix

	points=contour[:,0,:]

	D_matrix=np.ones((points.shape[0],6))
	D_matrix[:,0]= points[:,0]*points[:,0] #x2
	D_matrix[:,1]= points[:,0]*points[:,1] #xy
	D_matrix[:,2]= points[:,1]*points[:,1] #y2
	D_matrix[:,3:5]= points #x,y,1

	
	# w, v = np.linalg.eig( np.matmul(C_matrix_inv, np.matmul(np.linalg.transpose(D),D)))
	w, v = eig( C_matrix, np.matmul(np.transpose(D_matrix),D_matrix))

	#print(w)
	if np.all(np.isreal(w)) and np.all(np.isreal(v)) and np.all(np.isfinite(w)) and np.all(np.isfinite(v)) :
		maxi= np.argmax(w)
		Lam= w[maxi]

		u_vec= v[:,maxi]
		u_vec= u_vec.reshape((6,1))

		muu= np.sqrt(1/(np.matmul(np.transpose(u_vec),np.matmul(np.transpose(C_matrix),u_vec))))

		answer=muu*u_vec
		A=answer[0]
		B=answer[1]
		C=answer[2]
		D=answer[3]
		E=answer[4]
		F=answer[5]

		

		DENOM= B*B-4*A*C
		aa= np.sqrt(2*(A*E*E+C*D*D-B*D*E + DENOM*F)*((A+C)+np.sqrt( (A-C)*(A-C) + B*B)))
		bb= np.sqrt(2*(A*E*E+C*D*D-B*D*E + DENOM*F)*((A+C)-np.sqrt( (A-C)*(A-C) + B*B)))
		x0= (2*C*D-B*E)/DENOM
		y0= (2*A*E-B*D)/DENOM

		if(B==0):
			if(A<C):
				theta=0
			else:
				theta= 90
		else:
			theta=np.arctan2(C-A-np.sqrt( (A-C)*(A-C) + B*B),B)*180/(2*np.pi)


		#print(answer.shape)
		residuals= np.matmul(D_matrix,answer)
		print(residuals.shape)
		error= sum(residuals*residuals)/(residuals.shape[0]*residuals.shape[0]*residuals.shape[0])
		print(error)


		return (x0,y0),(aa,bb),theta, error
	else:
		return (0,0),(0,0),0, 99999


	#print(aa,bb,x0,y0,theta)


def splitContourUpDown(contour):
	#up and down
	index_left= np.argmin(contour[:,0,0])
	index_right= np.argmax(contour[:,0,0])

	indecies= np.sort(np.array([index_left,index_right])) #smallest to largest
	contour1= contour[indecies[0]:(indecies[1]+1),:,:]
	contour2=np.concatenate((contour[indecies[1]:,:,:], contour[:indecies[0]+1,:,:]), axis=0)
	return contour1,contour2

def splitContourLeftRight(contour):
	#left right
	index_up= np.argmin(contour[:,0,1])
	index_down= np.argmax(contour[:,0,1])

	indecies= np.sort(np.array([index_up,index_down])) #smallest to largest
	contour1= contour[indecies[0]:(indecies[1]+1),:,:]
	contour2=np.concatenate((contour[indecies[1]:,:,:], contour[:indecies[0]+1,:,:]), axis=0)
	return contour1,contour2






# root = Tkinter.Tk()
# root.withdraw()
# imgname = tkFileDialog.askopenfilename()

# print('trying to load image')
# print(imgname)
# img = cv2.imread(imgname)


dirpath = os.getcwd()

for subdir, dirs, files in os.walk(dirpath + '/Images/Night_100_Duo_selected'):
	files.sort()
	for file in files[3:]:
		filepath = subdir + os.sep + file
		if filepath.endswith(".jpg") or filepath.endswith(".pgm") or filepath.endswith(".png") or filepath.endswith(".ppm"):




			imgname=filepath
			# load image
			print('----------------------')
			print(file)
			img = cv2.imread(imgname)
			findcircles(img)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

# cimg = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)








cv2.waitKey(0)

# When everything done, release the capture
cv2.destroyAllWindows()