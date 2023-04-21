#Documentation: https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
import cv2 as cv
#print(cv.__version__)

img = cv.imread('src/Lena.jpeg')
#cv.imshow('original', img)

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('gray', img_gray)

img_gray_blur = cv.GaussianBlur(img_gray, (3,3), 0) #Fingure out which kernel size is best
cv.imshow('blurred', img_gray_blur)

img_sobel =  cv.Sobel(src=img_gray_blur, ddepth=cv.CV_64F, dx=1, dy=1, ksize=3)
cv.imshow('sobel', img_sobel)

img_canny = cv.Canny(image=img_gray_blur, threshold1=100, threshold2=200)
cv.imshow('canny', img_canny)

#Maybe implement non maximum suppression if sobel method is chosen

cv.waitKey()
print('success')