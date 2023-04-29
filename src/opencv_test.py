#Documentation: https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
import cv2 as cv
import numpy as np
import sys
#print(cv.__version__)

img = cv.imread('src/Tex1_square.jpg') 
#Size should be reduced for texile images
#cv.imshow('original', img)

down_width = 680
down_height = 420
down_points = (down_width, down_height)

resize_down = cv.resize(img, down_points, interpolation=cv.INTER_LINEAR)

#cv.imshow('down', resize_down)

img_gray = cv.cvtColor(resize_down, cv.COLOR_BGR2GRAY)
#cv.imshow('gray', img_gray)

img_gray_blur = cv.GaussianBlur(img_gray, (5,5), 0) #Fingure out which kernel size is best
#cv.imshow('blurred', img_gray_blur)

#img_sobel =  cv.Sobel(src=img_gray_blur, ddepth=cv.CV_64F, dx=1, dy=1, ksize=13)
#cv.imshow('sobel', img_sobel)

img_canny_blur = cv.Canny(image=img_gray_blur, threshold1=200, threshold2=275)
#cv.imshow('canny_blur', img_canny_blur)

img_canny_clean = cv.Canny(image=img_gray, threshold1=200, threshold2=275) # numpy.ndarray
#cv.imshow('canny_clean', img_canny_clean)

#with np.printoptions(threshold=sys.maxsize):
#    print(img_canny_clean)
np.savetxt('img_canny_clean.txt', img_canny_clean)

print('Length: ', len(img_canny_clean))
count = 0
count2 = 0
count3 = 0
flag = False
white_pixel_list = []
class pixel_class:
    x = 0
    y = 0
    val = 0
    vector_to_next_x = 0
    vector_to_next_y = 0

for i in range(len(img_canny_clean)):
    #print('i: ', len(i))
    count = count+1
#    print('count: ', count)
    count = 0
    for j in range(len(img_canny_clean[i])):
        #print('j: ', j)
        #print('X: ', img_canny_clean[i][j])
        count = count+1
        #print('count2: ', count)
        pixel = img_canny_clean[i][j]
        if pixel > 0:
            white_pixel = pixel_class()
            white_pixel.x = i
            white_pixel.y = j
            white_pixel.val = pixel
            white_pixel_list.append(white_pixel)
            #flag = True
            #print('pixel_list: ', white_pixel_list[0])
            #break
    if flag == True:
        break


vector_list = []

for i in range(len(white_pixel_list)): # error: making vector to next white pixel, which might be in next row/col
    if i < len(white_pixel_list)-1:
        point1 = white_pixel_list[i]
        point2 = white_pixel_list[i+1]
        vector = pixel_class
        vector.vector_to_next_x = point2.x - point1.x
        vector.vector_to_next_y = point2.y - point1.y
#        print('vector_to_next_x: ', vector.vector_to_next_x)
#        print('vector_to_next_y: ', vector.vector_to_next_y)
        vector_list.append(vector)
#        print('i: ', i)

for i in range(len(vector_list)):
    vector_len = np.sqrt(vector_list[i].vector_to_next_x**2 + vector_list[i].vector_to_next_y**2)
    print('vector_to_next_x: ', vector_list[i].vector_to_next_x)
    print('vector_to_next_y: ', vector_list[i].vector_to_next_y) # something is rotten here
    #print('vector len: ', vector_len) #seems unlikely that they are all length 1
    if vector_len > 5:
        vector_list.pop(i)


print('vector_list len: ', len(vector_list)) #=1634
print('pixel_list: ', len(white_pixel_list)) #=1635

#print('index: ', img_canny_clean.index(255))

#        for k in range(img_canny_clean[i][j]):
        #    pass
#            print('k: ', k)
#            count3 = count3+1
#            print('count3: ', count3)
        #if j == 255:
         #   print('white pixel')


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
#cv.imshow('img_combine', img_combine)

for i in range(2):
    img_combine = cv.morphologyEx(img_combine, cv.MORPH_ERODE, kernel)
#cv.imshow('img_combine_erode', img_combine)

#corner detection
img_canny_blur_float = np.float32(img_canny_blur)
corn0 = cv.cornerHarris(img_canny_blur_float, 2, 3, 0.04) 
corn0 = cv.dilate(corn0, None)
#cv.imshow('corn0', corn0) # best so far




img_canny_clean_float = np.float32(img_canny_clean)
corn = cv.cornerHarris(img_canny_clean_float, 2, 3, 0.04) 
corn = cv.dilate(corn, None)
#cv.imshow('corn', corn)

img_canny_clean_blur = cv.GaussianBlur(img_canny_clean, (21,21), 0) #Fingure out which kernel size is best
#cv.imshow('img_canny_clean_blur', img_canny_clean_blur)

img_canny_clean_blur_float = np.float32(img_canny_clean_blur)
corn2 = cv.cornerHarris(img_canny_clean_blur_float, 2, 3, 0.04) 
corn2 = cv.dilate(corn2, None)
#cv.imshow('corn2', corn2)



img_combine_float = np.float32(img_combine)
cornx = cv.cornerHarris(img_combine_float, 2, 3, 0.04) 
cornx = cv.dilate(cornx, None)
#cv.imshow('cornx', cornx)



#img_combine_float = np.float32(img_combine)
#dst = cv.cornerHarris(img_combine_float, 2, 3, 0.04)
#dst = cv.dilate(dst, None)



#cv.imshow('dst', dst)

#cv.imshow('corn', corn)



#Maybe implement non maximum suppression if sobel method is chosen



cv.waitKey()

#cv.imwrite('pics/tex1_canny.jpg', img_canny)

print('END')