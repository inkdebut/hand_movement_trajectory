# -*- coding: utf-8 -*-
# @Author  : 扪参历井仰胁息の蔡裕成
# @FileName: main.py
# @Date    : 2022/1/13 一月 2022
# @Time    : 11:21
# @Project : hand_movement_trajectory

#Warning:
#I wrote all these comments as Chinese, because the relationship between time and some reasons can not
#be changed one by one,you can choose to copy and then go to the relevant website or software translater


#基于cv2和mediapipe(介质媒体),检测摄像头所拍到的画面中的手，并画出关节连线与追踪其运动轨迹
#想修改检测的精度与运行速度以及摄像头所能检测到的手的数量可以在Hand()中修改参数


#导入包
import time
import cv2
import mediapipe as mp

#捕捉摄像头，如果只有一个则默认为0
cap = cv2.VideoCapture(0)

#调用手部函数
mpHands = mp.solutions.hands
hands = mpHands.Hands()

#画手部的线条
mpDraw = mp.solutions.drawing_utils

handlmsStyle=mpDraw.DrawingSpec(color=(255,255,255),thickness=5) #手部的关节点，颜色为白色，粗细设置为5
handconStyle=mpDraw.DrawingSpec(color=(0,255,0),thickness=3) #手部的线条，颜色为绿色，粗细设置为3

#帧率显示所需要的变量
pTime = 0    #present time  循环结尾设置pTime=cTime，然后在下一次循环中得出循环中的cTime-pTime，fps=1/cTime-pTime，即当前帧率
cTime = 0    #current time 当前时间，即每次循环所增加的时间

#主循环
while True:
    ret, img = cap.read()
    imgWidth = img.shape[1]
    imgHeight = img.shape[0]
    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #将颜色由BGR转为RGB
        result = hands.process(imgRGB)
        print(result.multi_hand_landmarks)  #手部的坐标
        if result.multi_hand_landmarks:
            for handlms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img,handlms,mpHands.HAND_CONNECTIONS,handlmsStyle,handconStyle)
                for i,lm in enumerate(handlms.landmark):

                    xPos = int(lm.x*imgWidth)
                    yPos = int(lm.y*imgHeight)
                    #打印手部关节点的数字，数字大小为0.3，颜色为红色
                    cv2.putText(img,str(i),(xPos-25,yPos+5),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,255),2)
                    #当i为4，即大拇指，点会放大，大小设定为10
                    if i==4:
                        cv2.circle(img,(xPos,yPos),10,(0,0,255),cv2.FILLED)
                    print(i,xPos,yPos)
        #设置帧率
        cTime =time.time()
        fps=1/(cTime-pTime)
        pTime =cTime
        #向当前屏幕输出打印字符串
        cv2.putText(img,f'FPS : {int(fps)}',(30,50),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),3)


        #显示摄像头窗口
        cv2.imshow('image', img)
        q = cv2.waitKey(100)
        if q == ord('q'):
            break

#常规操作，循环结束以后释放cap
cap.release()


