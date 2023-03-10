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
    #上限を1に正規化
    dst = np.copy(np.sqrt( dst_x**2 + dst_y**2 ))
    dst = dst / dst.max()

    #水平方向と垂直方向の微分画像の勾配の向きthetaを計算
    #[rad]から[deg]に変換
    theta = np.rad2deg( np.arctan2( dst_y,dst_x))

    return dst, theta #dst:gradient magnitude   theta:gradient direction

def CannyEdgeDetection(G,G_theta):
    
    height, width, color = G.shape[0], G.shape[1], G.shape[2]
    
    dst = np.zeros((height,width,color))
    G_dir = np.zeros((height,width)) #0[deg] 45[deg] 90[deg] 135[deg]

    #非極大値抑制(Non Maximum Suppression)#####################################################
    for i in range(height):
        for j in range(width):

            theta = G_theta[i,j,0]

            #勾配の向きを４方向に離散化

            #勾配の向きが0[deg]-> -22.5 <= theta < 22.5 or 157.5 <= theta < 202.5
            #x軸で分かれるので条件も分ける
            if  (-22.5 <= theta <= 22.5) or (-157.5 >= theta) or (theta >= 157.5):
                G_dir[i,j] = 0#[deg]

            #勾配の向きが45[deg]-> 22.5 <= theta < 67.5 or -157.5 <= theta < -112.5
            elif(22.5 < theta <= 67.5) or (-157.5 < theta <= -112.5):
                G_dir[i,j] = 45#[deg]

            #勾配の向きが90[deg]-> 67.5 <= theta < 112.5 or -112.5 <= theta < -67.5
            elif(67.5 < theta <= 112.5) or (-112.5 <= theta  < -67.5):
                G_dir[i,j] = 90#[deg]

            #勾配の向きが135[deg]-> 112.5 <= theta < 157.5 or -67.5 <= theta < -22.5
            elif(112.5 <= theta and theta < 157.5) or (-67.5 <= theta and theta < -22.5):
                G_dir[i,j] = 135#[deg]

    #端っこは処理できないので１を初期値
    for i in range(1,height-1):
        for j in range(1,width-1):
            if G_dir[i,j] == 0:               
                if ((G[i,j-1,0] > G[i,j,0]) | (G[i,j+1,0] > G[i,j,0])):
                    dst[i,j,:] = 0
                else:
                    dst[i,j,:] = G[i,j,0]
            elif G_dir[i,j] == 45:
                if ((G[i-1,j+1,0] > G[i,j,0]) | (G[i+1,j-1,0] > G[i,j,0])):
                    dst[i,j,:] = 0
                else:
                    dst[i,j,:] = G[i,j,0]
            elif G_dir[i,j] == 90:
                if ((G[i-1,j,0] > G[i,j,0]) | (G[i+1,j,0] > G[i,j,0])):
                    dst[i,j,:] = 0
                else:
                    dst[i,j,:] = G[i,j,0]
            elif G_dir[i,j] == 135:
                if ((G[i-1,j-1,0] > G[i,j,0]) | (G[i+1,j+1,0] > G[i,j,0])):
                    dst[i,j,:] = 0
                else:
                    dst[i,j,:] = G[i,j,0]

    #HysteresisThreshold処理######################################################################
    #エッジ群を強いエッジ弱いエッジ中間のエッジに３分割
    max_threshold = 0.2
    min_threshold = 0.03

    strong = 1
    weak = 0.5

    dst_threshold = np.zeros((height,width,color))
    dst_hysteresis = np.zeros((height,width,color))

    for i in range(height):
        for j in range(width):
            if max_threshold < dst[i,j,0]: #エッジの強い閾値
                dst_threshold[i,j,:] = strong
            elif min_threshold < dst[i,j,0]: #エッジの弱い閾値
                dst_threshold[i,j,:] = weak

    #ヒステリシスによるエッジ処理
    #端は処理できないので初期値は１
    for i in range(1,height-1):
        for j in range(1,width-1):
            if dst_threshold[i,j,0] == strong:
                dst_threshold[i,j,:] = 1
            elif dst_threshold[i,j,0] == weak:
                #弱いと判断されたエッジの周辺に強いエッジがあるか判定
                if (dst_threshold[i,j-1,0] == strong) | (dst_threshold[i,j+1,0] == strong) | (dst_threshold[i+1,j,0] == strong) | (dst_threshold[i-1,j,0] == strong) | (dst_threshold[i+1,j+1,0] == strong) | (dst_threshold[i+1,j-1,0] == strong) | (dst_threshold[i-1,j+1,0] == strong) | (dst_threshold[i-1,j-1,0] == strong):
                    dst_threshold[i,j,:] = 1
                else:
                    dst_threshold[i,j,:] = 0
    
    dst_hysteresis = np.copy(dst_threshold)
    return dst_hysteresis

def LineFunc(y,x,theta):
    theta = np.deg2rad(theta)
    return (x * np.cos(theta)) + (y * np.sin(theta))

def HoughTransform(proc_img):
    height, width, color = proc_img.shape[0], proc_img.shape[1], proc_img.shape[2]
    diagonal_distance = np.int(np.ceil(np.sqrt(height**2 + width**2)))
    rho_theta_count = np.zeros((diagonal_distance,180))

    for i in range(height):
        for j in range(width):
            if(proc_img[i,j,0] == 1):
                for theta in range(180):
                    rho = np.int(np.round(LineFunc(i,j,theta)))
                    rho_theta_count[rho,np.int(theta)] += 1

    rho_theta_count = rho_theta_count / rho_theta_count.max()
    return rho_theta_count

def ProtHoughTransformLine(proc_img,rho_theta):
    height, width, color = proc_img.shape[0], proc_img.shape[1], proc_img.shape[2]
    vote_max_index = np.unravel_index(np.argmax(rho_theta), rho_theta.shape)
    rho, theta = vote_max_index[0], vote_max_index[1]

    print(rho)
    print(theta)

    for i in range(width):
        y = np.int((rho / np.sin(np.deg2rad(theta))) - (i * (np.cos(np.deg2rad(theta)) / np.sin(np.deg2rad(theta)))))
        if (0 < y < height):
            proc_img[y,i,0] = 1
            proc_img[y,i,1] = 1
            proc_img[y,i,2] = 1

    for i in range(height):
        x = np.int((rho / np.cos(np.deg2rad(theta))) - (i * (np.sin(np.deg2rad(theta)) / np.cos(np.deg2rad(theta)))))
        if (0 < x < width):
            proc_img[i,x,0] = 1
            proc_img[i,x,1] = 1
            proc_img[i,x,2] = 1

    return proc_img
    
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
    hough_vote = HoughTransform(img_canny)
    img_hough = ProtHoughTransformLine(img,hough_vote)
    
    #文字入れ
    cv2.putText(img_gray,text='GrayscaleTransform',org=(10,30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,0,255),thickness=2,lineType=cv2.LINE_4)
    cv2.putText(img_gausian,text='GaussianFilter',org=(10,30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,0,255),thickness=2,lineType=cv2.LINE_4)
    cv2.putText(img_sobel,text='SobelFilter',org=(10,30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,0,255),thickness=2,lineType=cv2.LINE_4)
    cv2.putText(img_canny,text='CannyEdgeDetection',org=(10,30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,0,255),thickness=2,lineType=cv2.LINE_4)

    img_conv = np.hstack((img_gray,img_gausian,img_sobel,img_canny,img_hough))

    cv2.imshow('image_gray', img_conv)
    cv2.waitKey(0)
    cv2.imshow('hough', img_hough)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()