#!/usr/bin/env python3

import cv2
import numpy as np

img_file_pass = './images/DigitalBatteryCapacityChecker.bmp'

#Passで指定された画像を格納、元画像が大きいので１０分の１まで圧縮
img = cv2.imread(img_file_pass)
img = cv2.resize(img,dsize=None,fx=0.1,fy=0.1)

#画像の縦、横、色素数を変数に格納
height, width, color = img.shape
print('image size: ',height ,',', width,',',color)

def grayscale(proc_img):
    #引数として渡された画像をグレースケールに変換して出力する関数
    height, width, color = proc_img.shape
    img_gray = np.zeros((height,width,color))

    img_gray[:,:,0] = proc_img[:,:,0] * 0.229 / 255.0 + proc_img[:,:,1] * 0.587 / 255.0 + proc_img[:,:,2] * 0.114 / 255.0
    img_gray[:,:,1] = proc_img[:,:,0] * 0.229 / 255.0 + proc_img[:,:,1] * 0.587 / 255.0 + proc_img[:,:,2] * 0.114 / 255.0
    img_gray[:,:,2] = proc_img[:,:,0] * 0.229 / 255.0 + proc_img[:,:,1] * 0.587 / 255.0 + proc_img[:,:,2] * 0.114 / 255.0

    return img_gray

def GaussianFilter(proc_img):
    #引数として渡された画像にガウシアンフィルタをかけ出力する関数
    return 0

def main():
    global img, height, width, color

    img_gray = np.zeros((height,width,color))
    img_gray = grayscale(img)

    cv2.imshow('image_gray', img_gray)
    cv2.waitKey(3000)

if __name__ == "__main__":
    main()