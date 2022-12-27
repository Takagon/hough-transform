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

    theta = np.zeros((height,width))

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

    #水平方向と垂直方向の微分画像の勾配の大きさを計算
    #上限を１に正規化
    dst = np.copy(np.sqrt( dst_x**2 + dst_y**2 ))
    dst = dst / dst.max()

    #水平方向と垂直方向の微分画像の勾配の向きthetaを計算
    #[rad]から[deg]に変換
    theta = np.rad2deg( np.arctan2( dst_y,dst_x))

    return dst, theta

def CannyEdgeDetection(G,G_theta):
    
    height, width, color = G.shape[0], G.shape[1], G.shape[2]
    
    dst = np.zeros((height,width,color))
    G_dir = np.zeros((height,width)) #0[deg] 45[deg] 90[deg] 135[deg]

    #非極大値抑制(Non Maximum Suppression)
    for i in range(height):
        for j in range(width):

            theta = G_theta[i,j,0]

            #勾配の向きが0[deg]-> -22.5 <= theta < 22.5 or 157.5 <= theta < 202.5
            #x軸で分かれるので条件も分ける
            if  (-22.5 <= theta and theta <= 22.5) or (-157.5 >= theta) or (theta >= 157.5):
                G_dir[i,j] = 0#[deg]

            #勾配の向きが45[deg]-> 22.5 <= theta < 67.5 or -157.5 <= theta < -112.5
            elif(22.5 < theta and theta <= 67.5) or (-157.5 < theta and theta <= -112.5):
                G_dir[i,j] = 45#[deg]

            #勾配の向きが90[deg]-> 67.5 <= theta < 112.5 or -112.5 <= theta < -67.5
            elif(67.5 < theta and theta <= 112.5) or (-112.5 <= theta and theta < -67.5):
                G_dir[i,j] = 90#[deg]

            #勾配の向きが135[deg]-> 112.5 <= theta < 157.5 or -67.5 <= theta < -22.5
            elif(112.5 <= theta and theta < 157.5) or (-67.5 <= theta and theta < -22.5):
                G_dir[i,j] = 135#[deg]

    return dst

def main():
    #画像ファイルのパス指定
    img_file_pass = './images/BatteryChecker.bmp'

    #Passで指定された画像を格納、元画像が大きいので10分の1まで圧縮
    img = cv2.imread(img_file_pass)
    img = cv2.resize(img,dsize=None,fx=0.1,fy=0.1)

    #画像の縦、横、色素数を変数に格納  
    height, width, color = img.shape

    img_gray = np.zeros((height,width,color))
    img_gray = grayscale(img)
    img_gausian = GaussianFilter(img_gray)
    img_sobel,G_theta = SobelFilter(img_gausian)
    img_canny = CannyEdgeDetection(img_sobel,G_theta)

    #文字入れ
    cv2.putText(img_gray,text='GrayscaleTransform',org=(10,30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,0,255),thickness=2,lineType=cv2.LINE_4)
    cv2.putText(img_gausian,text='GaussianFilter',org=(10,30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,0,255),thickness=2,lineType=cv2.LINE_4)
    cv2.putText(img_sobel,text='SobelFilter',org=(10,30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,0,255),thickness=2,lineType=cv2.LINE_4)

    img_conv = np.hstack((img_gray,img_gausian,img_sobel))

    cv2.imshow('image_gray', img_conv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()