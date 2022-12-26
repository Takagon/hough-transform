import cv2
import numpy as np

img_file_pass = './images/DigitalBatteryCapacityChecker.bmp'

img = cv2.imread(img_file_pass)
img = cv2.resize(img,dsize=None,fx=0.1,fy=0.1)

cv2.imshow('image', img)
cv2.waitKey(3000)