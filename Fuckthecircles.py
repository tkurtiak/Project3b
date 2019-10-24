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



def findcenter(img):
	print('loaded image')
	#cv2.imshow('original image',img)

	
	#param2 is the threshold for circle detection
	whitedef=220
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


	cnts = cv2.findContours(255*white_mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	max_contour=None
	if len(cnts)>0:
		max_contour= max(cnts,key=cv2.contourArea)

	contour_img=0*white_mask.copy()
	if max_contour is not None:
		contour_img=cv2.drawContours(contour_img, [max_contour], -1, (255), -1)

	#cv2.imshow('the sheet',contour_img)
	rect = cv2.minAreaRect(max_contour)
	#print(rect)
	sides=np.sort(rect[1])
	AR= sides[1]/sides[0]
	print(AR)
	
	if 1.18<AR<1.72:
		area = cv2.contourArea(max_contour)
		hull = cv2.convexHull(max_contour)
		hull_area = cv2.contourArea(hull)
		solidity = float(area)/hull_area
		print(solidity)
		if solidity>.8:
			cv2.circle(imgOG,(int(rect[0][0]),int(rect[0][1])),3,(255,0,0),-1)

	cv2.imshow('output',imgOG)


	# edges = cv2.Canny(contour_img,int(.3*p1),p1)

	# cv2.imshow('edges',edges)

	# lines = cv2.HoughLines(edges,1,np.pi/180, 40) 

	# if lines is not None:
	# 	print('lines')
	# 	lines=lines[:,0,:]
	# 	print(lines.shape)
	# 	# print(lines)
	# 	if (lines.shape[0])>3:
			
	# 		#put these lines into 2 groups
	# 		# angles= lines[:,1].reshape((lines.shape[0], 1))
	# 		# print(angles.shape)
	# 		angles,labels=kmeans2(lines,4,iter=5,minit='points')

	# 		print(angles)
	# 		print(labels)
	# 		draw_lines(lines,img.copy(),labels)


	# edges_Fat=edges.copy()
	# edges_Fat = cv2.dilate(edges_Fat,np.ones((3,3), np.uint8),iterations=2)
	# edges_Fat = cv2.erode(edges_Fat,np.ones((3,3), np.uint8),iterations=1)

	# cv2.imshow('edges_Fat',edges_Fat)

	

	# cnts = cv2.findContours(edges_Fat, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	# cnts = imutils.grab_contours(cnts)

	

	# 


	




# root = Tkinter.Tk()
# root.withdraw()
# imgname = tkFileDialog.askopenfilename()

# print('trying to load image')
# print(imgname)
# img = cv2.imread(imgname)


dirpath = os.getcwd()

for subdir, dirs, files in os.walk(dirpath + '/Images/Night_100_Duo_selected'):
	files.sort()
	for file in files:
		filepath = subdir + os.sep + file
		if filepath.endswith(".jpg") or filepath.endswith(".pgm") or filepath.endswith(".png") or filepath.endswith(".ppm"):




			imgname=filepath
			# load image
			print(file)
			img = cv2.imread(imgname)
			findcenter(img)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

# cimg = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)








cv2.waitKey(0)

# When everything done, release the capture
cv2.destroyAllWindows()