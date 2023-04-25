#Documentation: https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
import cv2 as cv
import numpy as np
#print(cv.__version__)

img = cv.imread('src/Tex6.jpg') 
#Size should be reduced for texile images
#cv.imshow('original', img)

down_width = 680
down_height = 420
down_points = (down_width, down_height)

resize_down = cv.resize(img, down_points, interpolation=cv.INTER_LINEAR)

#cv.imshow('down', resize_down)

img_gray = cv.cvtColor(resize_down, cv.COLOR_BGR2GRAY)
#cv.imshow('gray', img_gray)

img_gray_blur = cv.GaussianBlur(img_gray, (3,3), 0) #Fingure out which kernel size is best
cv.imshow('blurred', img_gray_blur)

#img_sobel =  cv.Sobel(src=img_gray_blur, ddepth=cv.CV_64F, dx=1, dy=1, ksize=13)
#cv.imshow('sobel', img_sobel)

img_canny_blur = cv.Canny(image=img_gray_blur, threshold1=200, threshold2=275)
cv.imshow('canny_blur', img_canny_blur)

img_canny_clean = cv.Canny(image=img_gray, threshold1=200, threshold2=275)
cv.imshow('canny_clean', img_canny_clean)

####

kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))

img_canny_clean_dilate = img_canny_clean

for i in range(2):
    img_canny_clean_dilate = cv.morphologyEx(img_canny_clean_dilate, cv.MORPH_DILATE, kernel)
#cv.imshow('img_canny_clean_dilate', img_canny_clean_dilate)

img_canny_blur_dilate = img_canny_blur

for i in range(14):
    img_canny_blur_dilate = cv.morphologyEx(img_canny_blur_dilate, cv.MORPH_DILATE, kernel)
#cv.imshow('img_canny_blur_dilate', img_canny_blur_dilate)


img_combine = img_canny_clean_dilate - cv.bitwise_not(img_canny_blur_dilate)
cv.imshow('img_combine', img_combine)

kernel2 = cv.getStructuringElement(cv.MORPH_RECT, (5,5))

for i in range(2):
    img_combine = cv.morphologyEx(img_combine, cv.MORPH_ERODE, kernel2)
cv.imshow('img_combine_erode', img_combine)

#corner detection



#Maybe implement non maximum suppression if sobel method is chosen



cv.waitKey()

#cv.imwrite('pics/tex1_canny.jpg', img_canny)

print('success')