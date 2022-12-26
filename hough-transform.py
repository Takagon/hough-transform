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
    height, width, color = proc_img.shape[0], proc_img.shape[1], proc_img.shape[2]
    img_gray = np.zeros((height,width,color))

    img_gray[:,:,0] = proc_img[:,:,0] * 0.229 / 255.0 + proc_img[:,:,1] * 0.587 / 255.0 + proc_img[:,:,2] * 0.114 / 255.0
    img_gray[:,:,1] = proc_img[:,:,0] * 0.229 / 255.0 + proc_img[:,:,1] * 0.587 / 255.0 + proc_img[:,:,2] * 0.114 / 255.0
    img_gray[:,:,2] = proc_img[:,:,0] * 0.229 / 255.0 + proc_img[:,:,1] * 0.587 / 255.0 + proc_img[:,:,2] * 0.114 / 255.0

    return img_gray

def GaussianFilter(proc_img):
    #引数として渡された画像にガウシアンフィルタをかけ出力する関数

    height, width, color = proc_img.shape[0], proc_img.shape[1], proc_img.shape[2]
    dst = np.zeros((height,width,color))

    gaussian_kernel = np.array([[1/16,2/16,1/16],
                                [2/16,4/16,2/16],
                                [1/16,2/16,1/16]])
    
    img_padding = np.zeros((height+2,width+2,color))
    img_padding[1:height+1,1:width+1,:] = np.copy(proc_img)

    #ガウシアンカーネルと画像を畳み込み演算
    max = 0.0
    for i in range(height):
        for j in range(width):
            dst[i,j,:] = np.sum(img_padding[i:i+3,j:j+3] * gaussian_kernel)
        
        if(max < dst[i,j,0]):#画像の画素の中で一番輝度の高い値を取得
            max = dst[i,j,0]

    #このままでは白飛びする。上限を１に正規化
    return dst/max

def main():
    global img, height, width, color

    img_gray = np.zeros((height,width,color))
    img_gray = grayscale(img)

    img_gray = GaussianFilter(img_gray)

    cv2.imshow('image_gray', img_gray)
    cv2.waitKey(3000)

if __name__ == "__main__":
    main()