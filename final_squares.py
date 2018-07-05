# Code to detect Green and Orange color patches and send the coordinates in Polar form
import numpy as np
import imutils
import cv2
import math
from math import atan 
import serial
import time

# Communicate with serial of Arduino

ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0)

# Capture image

cam=cv2.VideoCapture(1)
cv2.namedWindow("Capture Image")
img_counter=0

while True:
	ret,frame=cam.read()
	cv2.imshow('frame',frame)
	if not ret:
	        break
	k = cv2.waitKey(1)

	if k%256 == 27:
	    print("Escape hit, closing...")
	    break
	elif k%256 == 32:
	    img_name = "lights_on_opencv_frame_{}.png".format(img_counter)
	    cv2.imwrite(img_name, frame)
	    print("{} written!".format(img_name))
	    img_counter += 1

cam.release()
cv2.destroyAllWindows()

# Load image

image=cv2.imread('lights_on_opencv_frame_{}.png', format(img_counter))
X,Y,_=np.shape(image)

#Image thresholding

boundaries=[([0,40,40],[50,85,88]),		# Green : Calibrate before using
			([0,30,79],[45,70,135])]	# Orange : Calibrate before using

count=0
green_x,green_y=0,0
orange_x,orange_y=0,0

for (lower, upper) in boundaries:
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
 
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)
 
	cv2.imshow("images_output", np.hstack([image, output]))
	cv2.waitKey(0)
	gray=cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	thresh = cv2.threshold(blurred, 35, 255, cv2.THRESH_BINARY)[1]

	cv2.imshow("images_thresh",thresh)
	cv2.waitKey(0)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]

	for c in cnts:
		M = cv2.moments(c)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
	 	
	 	if count==0:
	 		green_x=cX
	 		green_y=cY
	 	if count==1:
	 		orange_x=cX
	 		orange_y=cY

		count=count+1
	 	# cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
		cv2.circle(image, (cX, cY), 7, (255, 255, 255), 1)
		# cv2.putText(image, "center", (cX - 20, cY - 20),
		# 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
	 
		cv2.imshow("Image_Result", image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

orange_x,orange_y=orange_x-green_x,green_y-orange_y
green_x,green_y=0,0

#Convert pixel values to real distance

ratio=0.1625
s=10	# calibrate before using
n=25.5	# calibrate before using

real_x=(int)(orange_x*ratio*-1)
real_y=(int)(orange_y*ratio*-1)
flag_x,flag_y='0','0'


theta_x0=0*180/math.pi
theta_y0=atan((n-s+20)/50.0)*180/math.pi

theta_x=atan(real_x/50.0)*180/math.pi
theta_y=atan((n-s+real_y+20)/50.0)*180/math.pi

print("Real_X=",real_x)
print("Real_Y=",real_y)

ser.write(abs(real_x))
ser.write(abs(real_y))

if real_x<0:
	ser.write('1')
	flag_x=1
else:
	ser.write('0')
	flag_x=0
if real_y<0:
	ser.write('1')
	flag_y=1
else:
	ser.write('0')
	flag_y=0

print("Flag_X=",flag_x)
print("Flag_Y=",flag_y)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
