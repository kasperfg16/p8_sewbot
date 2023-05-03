import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def find_file(filename, search_path):
    """
    Search for a file in the given search path.
    Returns the full path to the file if found, or None otherwise.
    """
    
    for root, dir, files in os.walk(search_path):
        if filename in files:
            return os.path.abspath(os.path.join(root, filename))
    return None

filename1 = "test{1}.png"
filename2 = "moved.png"
search_path = "./"
model_path1 = find_file(filename1, search_path)
model_path2 = find_file(filename2, search_path)
backgr = cv2.imread(model_path1)
moved = cv2.imread(model_path2)
new = cv2.subtract(backgr, moved)

imgray = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)

edged = cv2.Canny(imgray, 30, 200)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

new = cv2.drawContours(new, contours, -1, (0,255,0), 3)
cv2.imshow('window',new)
# De-allocate any associated memory usage  
if cv2.waitKey(0) & 0xff == 27: 
    cv2.destroyAllWindows()