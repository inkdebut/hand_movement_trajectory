# -*- coding: utf-8 -*-
# @Author : Cai Yucheng
# @FileName: main.py
# @Date : 1/13/2022 January 2022
# @Time : 11:21
# @Project : hand_movement_trajectory


#based on cv2 and mediapipe (medium media), detect the hand in the picture captured by the camera, and draw the joint connection and track its movement trajectory
#if you want to modify the accuracy and running speed of the detection and the number of hands that the camera can detect, you can modify the parameters in Hand()


#import package
import time
import cv2
import mediapipe as mp

#capture the camera, if there is only one, the default is 0
cap = cv2.VideoCapture(0)

#call hand function
mpHands = mp.solutions.hands
hands = mpHands.Hands()

#draw the lines of the hands
mpDraw = mp.solutions.drawing_utils

handlmsStyle=mpDraw.DrawingSpec(color=(255,255,255),thickness=5) #the joint point of the hand, the color is white, and the thickness is set to 5
handconStyle=mpDraw.DrawingSpec(color=(0,255,0),thickness=3) #the line of the hand, the color is green, and the thickness is set to 3

#frame rate display required variables
pTime = 0 #present time Set pTime=cTime at the end of the loop, and then get the cTime-pTime in the loop in the next loop, fps=1/cTime-pTime, which is the current frame rate
cTime = 0 #current time The current time, that is, the time added by each loop

#main loop
while True:
    ret, img = cap.read()
    imgWidth = img.shape[1]
    imgHeight = img.shape[0]
    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert the color from BGR to RGB
        result = hands.process(imgRGB)
        print(result.multi_hand_landmarks) #the coordinates of the hand
        if result.multi_hand_landmarks:
            for handlms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img,handlms,mpHands.HAND_CONNECTIONS,handlmsStyle,handconStyle)
                for i,lm in enumerate(handlms.landmark):

                    xPos = int(lm.x*imgWidth)
                    yPos = int(lm.y*imgHeight)
                    #print the number of hand joint points, the number size is 0.3, and the color is red
                    cv2.putText(img,str(i),(xPos-25,yPos+5),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,255),2)
                    #when i is 4, that is, the thumb, the point will be enlarged, and the size is set to 10
                    if i==4:
                        cv2.circle(img,(xPos,yPos),10,(0,0,255),cv2.FILLED)
                    print(i,xPos,yPos)
        #set frame rate
        cTime = time.time()
        fps=1/(cTime-pTime)
        pTime = cTime
        #output the print string to the current screen
        cv2.putText(img,f'FPS: {int(fps)}',(30,50),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),3)


        # show the camera window
        cv2.imshow('image', img)
        q = cv2.waitKey(100)
        if q == ord('q'):
            break

#regular operation, release cap after the loop ends
cap.release()
