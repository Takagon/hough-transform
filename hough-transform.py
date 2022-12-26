import cv2
import numpy as np

img_file_pass = './images/DigitalBatteryCapacityChecker.bmp'
#img_file_pass = './images/Airplane.bmp'

img = cv2.imread(img_file_pass)
img = cv2.resize(img,dsize=None,fx=0.2,fy=0.2)

height, width, color = img.shape
print('image size: ',height ,',', width,',',color)

img_gray = np.zeros((height,width,color))
img_gray1d = np.zeros((height,width))
img_2gray = np.zeros((height,width,color))
img_sobel = np.zeros((height-2,width-2,color))

# roberts_mask_y = np.array([[ 1, 0],
#                            [ 0,-1]])

# roberts_mask_x = np.array([[ 0, 1],
#                            [-1, 0]])

sobel_mask_y =   np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])

sobel_mask_x =   np.array([[-1,-2,-1],
                           [ 1, 0, 0],
                           [ 1, 2, 1]])

def grayscale():
    global img, img_gray
    global height, width, color

    v = img[:,:,0] * 0.229 / 255.0 + img[:,:,1] * 0.587 / 255.0 + img[:,:,2] * 0.114 / 255.0
    #v = img[:,:,0] * 0.333 / 255.0 + img[:,:,1] * 0.333 / 255.0 + img[:,:,2] * 0.333 / 255.0

    img_gray[:,:,0] = v
    img_gray[:,:,1] = v
    img_gray[:,:, 2] = v
    img_gray1d[:,:] = v

    return 0

def threshold():
    global img_2gray,img,img_gray
    global height, width, color
    t = 0.5
    img_2gray[:,:,0] = np.where(img_gray1d > t, 1, 0)
    img_2gray[:,:,1] = np.where(img_gray1d > t, 1, 0)
    img_2gray[:,:,2] = np.where(img_gray1d > t, 1, 0)
    return 0

def prosses_sobel_x():
    global img_2gray,img,img_gray,img_sobel
    global height, width, color
    global sobel_mask_x,sobel_mask_y

    img_temp = np.zeros((height,width))

    for i in range(1,height-1,1):
        for j in range(1,width-1,1):
            v = img_gray[i-1,j-1,0]*sobel_mask_x[0,0] + img_gray[i-1,j,0]*sobel_mask_x[0,1] + img_gray[i-1,j+1,0]*sobel_mask_x[0,2] + img_gray[i,j-1,0]*sobel_mask_x[1,0] + img_gray[i,j,0]*sobel_mask_x[1,1] + img_gray[i,j+1,0]*sobel_mask_x[1,2] + img_gray[i+1,j-1,0]*sobel_mask_x[2,0] + img_gray[i+1,j,0]*sobel_mask_x[2,1] + img_gray[i+1,j+1,0]*sobel_mask_x[2,2]
            img_sobel[i-1,j-1,0] = v
            img_sobel[i-1,j-1,1] = v
            img_sobel[i-1,j-1,2] = v
    return 0

def prosses_sobel_y():
    global img_2gray,img,img_gray,img_sobel
    global height, width, color
    global sobel_mask_x,sobel_mask_y
    return 0

def prosses_roberts_x():
    return 0

def prosses_roberts_y():
    return 0

def main():
    global img, img_2gray, img_sobel
    grayscale()
    prosses_sobel_x()
    # cv2.imshow('image', img)
    cv2.imshow('image_gray', img_sobel)
    cv2.waitKey(3000)

if __name__ == "__main__":
    main()