import cv2
import numpy as np
import pandas as pd

index = ['color','color_name','hex','R','G','B']
data = pd.read_csv("colors.csv",names=index,header=None)

# Finding Color of the centroid of Contours
def get_color_name(r,g,b):
    minimum = 10000
    for i in range(len(data)):
        d = abs(r - int(data.loc[i,'R'])) + abs(g - int(data.loc[i,'G'])) + abs(b - int(data.loc[i,"B"]))
        if d<=minimum:
            minimum = d
            cname = data.loc[i,"color_name"]
    return cname



img = cv2.imread("images/tri3.png")
cv2.imshow("Original",img)

# GrayScale
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Blur the Image 
blurred = cv2.GaussianBlur(gray_img,(3,3),cv2.BORDER_DEFAULT)

# Detect the Edges 
canny = cv2.Canny(blurred,110,175)

# Dilation 
dilation = cv2.dilate(canny,(3,3),iterations=2)

# Eroding
erode = cv2.erode(dilation,(3,3),iterations=1)

# Finding Contour
contours, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    M = cv2.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    centroid = (cx,cy)
    b,g,r = img[cx,cy]
    r,g,b = int(r),int(g),int(b)
    color = get_color_name(r,g,b)

    approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
    cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5
    if len(approx) == 3:
        shape = "Triangle"
    elif len(approx) == 4 :
        shape = "Quadrilateral"
    elif len(approx) == 5 :
        shape = "Pentagon"
    elif len(approx) == 6: 
        shape = "Hexagon"
    elif len(approx) == 10 :
        shape = "Star"
    else:
        shape = "Circle"
    cv2.putText(img, f"{shape} - {color}", (x-50, y+130), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0))


cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()