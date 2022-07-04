import cv2
import numpy as np

def stackImages(imgArray,scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor

    return ver


def rectContours(contours) :
    rectCon = []
    for i in contours :
        area= cv2.contourArea(i)
        if area>50 :
           peri = cv2.arcLength(i,True)
           approx= cv2.approxPolyDP(i,0.02*peri,True)
           if len(approx)==4 :
               rectCon.append(i)
    rectCon= sorted(rectCon,key=cv2.contourArea,reverse=True)

    return rectCon


def getCornerContours(cont) :
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True)
    return  approx

def reorder(points) :
    points = points.reshape((4,2))
    newPoints = np.zeros((4,1,2), np.int32)
    add= points.sum(1)
    newPoints[0] = points [np.argmin(add)]
    newPoints[3] = points[np.argmax(add)]
    newPoints[1] = points [3]
    newPoints[2] = points[1]
    return newPoints

def spliBoxes(img) :
    rows= np.vsplit(img,5)
    boxes = []
    for r in rows :
        colms= np.hsplit(r,5)
        for box in colms :
            boxes.append(box)
    return  boxes

def showAnswers(img,myIndex,grading,ans) :
    secW = int(img.shape[1]/5)
    secH = int(img.shape[0]/5)
    for x in range(0,5) :
        myAns = myIndex[x]
        cX=(myAns*secW)+secW//2
        cY=(x*secH)+secH//2
        if grading[x]==1:
            mycolor=(0,255,0)
        else:
            mycolor=(0,0,250)
            cv2.circle(img, ( (ans[x] * secW) + secW // 2, (x * secH) + secH // 2), 20, (0,255,0), cv2.FILLED)

        cv2.circle(img,(cX,cY),50,mycolor,cv2.FILLED)

    return img






