import cv2
import numpy as np
from matplotlib import pyplot as plt

def getEdges(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),cv2.BORDER_DEFAULT)
    canny = cv2.Canny(blur,100,175)
    return canny

# Drawing Lines 
def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image


# Area of interest
def roi(image):
    # Triangle Image 
    height = image.shape[0]
    triangle = np.array([[(220,height),(1100,height),(550,250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,triangle,255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

image = cv2.imread("lane.jpg")

edges = getEdges(image)
region = roi(edges)
lines = cv2.HoughLinesP(region,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
line_image = display_lines(image,lines)

lane_image = cv2.addWeighted(image,0.8,line_image,1,1)
cv2.imshow("Lane Detection",lane_image)
cv2.waitKey(0)
cv2.destroyAllWindows()