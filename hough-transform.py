#!/usr/bin/env python3

import cv2
import numpy as np

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

    #ガウシアンフィルタのカーネルを定義
    gaussian_kernel = np.array([[1/16,2/16,1/16],
                                [2/16,4/16,2/16],
                                [1/16,2/16,1/16]])
    
    #畳み込み演算できるように周囲に１ピクセル分追加する
    img_padding = np.zeros((height+2,width+2,color))
    img_padding[1:height+1,1:width+1,:] = np.copy(proc_img)

    #ガウシアンカーネルと画像を畳み込み演算
    brightness_max = 0.0
    for i in range(height):
        for j in range(width):
            dst[i,j,:] = np.sum(img_padding[i:i+3,j:j+3,0] * gaussian_kernel)
        
            if(brightness_max < dst[i,j,0]):#画像の画素の中で一番輝度の高い値を取得
                brightness_max = dst[i,j,0]

    return dst/brightness_max #このままでは白飛びする。上限を１に正規化

def SobelFilter(proc_img):
    #引数として渡された画像にソーベルフィルターをかけ出力する関数
    height, width, color = proc_img.shape[0], proc_img.shape[1], proc_img.shape[2]

    dst = np.zeros((height,width,color))
    dst_x = np.zeros((height,width,color))
    dst_y = np.zeros((height,width,color))

    #ソーベルフィルタの垂直方向と水平方向のカーネルを定義
    sobel_x_kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])

    sobel_y_kernel = np.array([[-1,-2,-1],
                               [ 0, 0, 0],
                               [ 1, 2, 1]])

    #畳み込み演算できるように周囲に１ピクセル分追加する
    img_padding = np.zeros((height+2,width+2,color))
    img_padding[1:height+1,1:width+1,:] = np.copy(proc_img)

    #水平方向のソーベルフィルタと画像の畳込み演算
    for i in range(height):
        for j in range(width):
            dst_x[i,j,:] = np.sum(img_padding[i:i+3,j:j+3,0] * sobel_x_kernel)

    #垂直方向のソーベルフィルタと画像の畳込み演算
    for i in range(height):
        for j in range(width):
            dst_y[i,j,:] = np.sum(img_padding[i:i+3,j:j+3,0] * sobel_y_kernel)

    #垂直方向のソーベルフィルタ画像と垂直方向のソーベルフィルタを合わせる
    dst = np.copy(np.sqrt( dst_x**2 + dst_y**2 ))
    return dst

def CannyEdgeDetection():
    return 0

def main():
    #画像ファイルのパス指定
    img_file_pass = './images/DigitalBatteryCapacityChecker.bmp'

    #Passで指定された画像を格納、元画像が大きいので１０分の１まで圧縮
    img = cv2.imread(img_file_pass)
    img = cv2.resize(img,dsize=None,fx=0.1,fy=0.1)

    #画像の縦、横、色素数を変数に格納  
    height, width, color = img.shape
    print('image size: ',height ,',', width,',',color)

    img_gray = np.zeros((height,width,color))
    img_gray = grayscale(img)
    img_gausian = GaussianFilter(img_gray)
    img_sobel = SobelFilter(img_gausian)

    img_conv = np.hstack((img,img_gray,img_gausian,img_sobel))

    cv2.imshow('image_gray', img_conv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()