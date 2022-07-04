import cv2
import numpy as np
import func

path = "qcm10.jpeg"
widthImg=700
heightImg=700
ans = [ 1,2,0,1,4]
webcamFeed = False

cap = cv2.VideoCapture(0)
cap.set(10,150)
while True :
    if webcamFeed : success ,img= cap.read()
    else:  img = cv2.imread(path)

    img=cv2.resize(img,(widthImg,heightImg))
    imgContours = img.copy()
    imgBiggestContours = img.copy()
    imgfinal=img.copy()

    imgGray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)

    imgCanny = cv2.Canny(imgBlur,10,50)


    try :
        contours,hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(imgContours,contours,-1,(0,250,0),10)
        rectCon = func.rectContours(contours)
        biggestContours = func.getCornerContours(rectCon[0])
        func.reorder(biggestContours)
        secondBigContours = func.getCornerContours(rectCon[1])


        if biggestContours.size != 0 and secondBigContours.size != 0 :
           cv2.drawContours(imgBiggestContours,biggestContours,-1,(0,255,0),20)
           cv2.drawContours(imgBiggestContours,secondBigContours,-1,(255,0,0),20)

           biggestContours = func.reorder(biggestContours)

           p1 = np.float32(biggestContours)
           p2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
           matrix = cv2.getPerspectiveTransform(p1, p2)
           imgwarp = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

           secondBigContours = func.reorder(secondBigContours)

           pg1 = np.float32(secondBigContours)
           pg2 = np.float32([[0,0],[325,0],[0,150],[325,150]])
           matrixg = cv2.getPerspectiveTransform(pg1, pg2)
           imgwarpg = cv2.warpPerspective(img, matrixg, (325, 150 ))



           imgwarpGray = cv2.cvtColor(imgwarp,cv2.COLOR_BGR2GRAY)
           imgtresh= cv2.threshold(imgwarpGray,170,255,cv2.THRESH_BINARY_INV)[1]


           boxes = func.spliBoxes(imgtresh)

           mypixelval= np.zeros((5,5))
           b =0
           for i in range (0,5):
               for j in range(0,5) :
                  mypixelval[i][j] = cv2.countNonZero(boxes[b])
                  b+=1

           myIndex = []
           for x in range(0,5) :
                arry= mypixelval[x]
                myIndexVal = np.where(arry == np.amax(arry))

                myIndex.append(myIndexVal[0][0])

           grading=[]
           for x in range(0,5) :
               if ans[x]== myIndex[x] :
                   grading.append(1)
               else: grading.append(0)

           score = (sum(grading)/5) *100

           imgAnsr=imgwarp.copy()
           imgAnsr=func.showAnswers(imgAnsr, myIndex, grading, ans)


           imgRawDrawing=np.zeros_like(imgwarp)
           imgRawDrawing=func.showAnswers(imgRawDrawing, myIndex, grading, ans)

           invmatrix = cv2.getPerspectiveTransform(p2, p1)
           imginvwarp = cv2.warpPerspective(imgRawDrawing, invmatrix, (widthImg, heightImg))

           imgRawGrade = np.zeros_like(imgwarpg)
           cv2.putText(imgRawGrade,str(int(score))+"%",(60,100),cv2.FONT_HERSHEY_COMPLEX,3,(0,250,250),3)

           invmatrixg = cv2.getPerspectiveTransform(pg2, pg1)
           imginvwarpg = cv2.warpPerspective(imgRawGrade, invmatrixg , (widthImg, heightImg))

           imgfinal= cv2.addWeighted(imgfinal,1,imginvwarp,1,0)
           imgfinal= cv2.addWeighted(imgfinal,1,imginvwarpg,1,0)



           imgBlank=np.zeros_like(img)
           imgArray=([img,imgGray,imgBlur,imgCanny],[imgContours,imgBiggestContours,imgwarp,imgtresh],
                     [imgAnsr,imgRawDrawing,imginvwarp,imgfinal])
    except:
        imgBlank=np.zeros_like(img)
        imgArray=([imgBlank,imgBlank,imgBlank,imgBlank],[imgBlank,imgBlank,imgBlank,imgBlank],
                     [imgBlank,imgBlank,imgBlank,imgBlank])

    imgStacked=func.stackImages(imgArray,0.3)

    cv2.imshow("Final image",imgfinal)
    cv2.imshow("All image",imgStacked)
    # ...
    if cv2.waitKey(1) & 0xFF ==  ord("s") :
        cv2.imwrite("FinalResult.jpg",imgfinal)
        cv2.waitKey(300)
