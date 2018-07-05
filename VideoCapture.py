# Starter Code to capture and store image
import numpy as np
import cv2

cam=cv2.VideoCapture(1)
cv2.namedWindow("Capture Image")

while True:
	ret,frame=cam.read()
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	cv2.imshow('frame',gray)
	if not ret:
	        break
	k = cv2.waitKey(1)

	if k%256 == 27:
	    print("Escape hit, closing...")
	    break
	elif k%256 == 32:
	    img_name = "opencv_frame_0.png"
	    cv2.imwrite(img_name, frame)

cam.release()
cv2.destroyAllWindows()