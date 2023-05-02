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
#    print(img_canny_clean)print('index: ', vector_list.index(123))
#np.savetxt('img_canny_clean.txt', img_canny_clean)

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
    corner_bool = False
#    vector_len = np.sqrt(vector_to_next_x**2 + vector_to_next_y**2)


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
print('Creating Vectors, please wait')
for i in range(len(white_pixel_list)):
    for j in range(len(white_pixel_list)):
        if i != j:
            point1 = white_pixel_list[i]
            point2 = white_pixel_list[j]
            vector = pixel_class()
            vector.x = white_pixel_list[i].x
            vector.y = white_pixel_list[i].y
            vector.vector_to_next_x = point2.x - point1.x
            vector.vector_to_next_y = point2.y - point1.y
            vector_len = np.sqrt(vector.vector_to_next_x**2 + vector.vector_to_next_y**2)
            if vector_len <= 1: #This number can be changed for faster but maybe less precise corners 
                vector_list.append(vector)


####################

def unit_vector(vector):
    #""" Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1x, v1y, v2x, v2y):
#    """ Returns the angle in radians between vectors 'v1' and 'v2'::

#            >>> angle_between((1, 0, 0), (0, 1, 0))
#            1.5707963267948966
#            >>> angle_between((1, 0, 0), (1, 0, 0))
#            0.0
#            >>> angle_between((1, 0, 0), (-1, 0, 0))
#            3.141592653589793
#    """
    v1 = (v1x, v1y)
    v2 = (v2x, v2y)
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

#####################3

#might have to remove redundency vectors for speed

print('vector_list len: ', len(vector_list)) #=2671590 now 13922(better)

print('Calculating angles for corners, hold on!')
interest_point_list = []
for i in range(len(vector_list)): # Best:4:21 Prev:4:38
    #print('Still Calculating... Round: ', i, 'of ', len(vector_list))
    for j in range(len(vector_list)):
        if i != j:
            vector1x = vector_list[i].vector_to_next_x
            vector1y = vector_list[i].vector_to_next_y
            vector2x = vector_list[j].vector_to_next_x
            vector2y = vector_list[j].vector_to_next_y
            angle = angle_between(vector1x, vector1y, vector2x, vector2y)
        
            if angle >= np.deg2rad(80) and angle <= np.deg2rad(100): # or angle <= np.deg2rad(-80) and angle >= np.deg2rad(-100):
                if vector_list[i].x == vector_list[j].x and vector_list[i].y == vector_list[j].y:
                    #print('angle: ', np.rad2deg(angle))        
                    #print('vector1x: ', vector_list[i].x)
                    #print('vector1y: ', vector_list[i].y)
                    #print('vector2x: ', vector_list[j].x)
                    #print('vector2y: ', vector_list[j].y)
                
                    interest_point_list.append(vector_list[i])
                    break
            

        #First check for othrogonaltity
        #use start of vector for corner point
#point_in_list = []
point_list_x = []
point_list_y = []

for i in range(len(interest_point_list)):
#    point_in_list = []
#    point_in_list.append(interest_point_list[i].x)
#    point_in_list.append(interest_point_list[i].y)
    point_list_x.append(interest_point_list[i].x)
    point_list_y.append(interest_point_list[i].y)

print('point_listx: ', point_list_x)
print('point_listy: ', point_list_y)

print('Drawing corners')
img_corner = img_canny_clean.copy() # Copy img_canny_clean into img_corner
for i in range(len(img_canny_clean)):
    for j in range(len(img_canny_clean[i])):
        img_corner[i][j] = 0
        if i in point_list_x and j in point_list_y: # kan man ikke. lige nu kigger den på om der er en x-værdi der passer anywhere i listen, men den passer ikke nødvedigvis med y-værdien. dumt lavet
            img_corner[i][j] = 255

#cv.imshow('img_corner', img_corner)

print('interest_point_list len: ', len(interest_point_list)) #= 8920 at range 3
print('vector_list len: ', len(vector_list)) #=2671590 now 13922(better)
print('pixel_list: ', len(white_pixel_list)) #=1635
#print('vector_list_reduced len: ', len(vector_list_reduced)) #=836 

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