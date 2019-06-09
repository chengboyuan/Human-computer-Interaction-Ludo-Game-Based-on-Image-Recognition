import pygame,sys
from pygame.locals import *
from random import randint
import cv2
import numpy as np
import copy
import math
import random

pygame.init()

BoardWidth = 1000
BoardHeight = 700
BPadding = 10
outlineWidth = 5

title = 'Ludo Board'

boxwidth = 50
boxheight = 50

WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
YELLOW = (255,255,0)
PURPLE = (255,20,255)
GREY = (184,184,184)

dispsurf = pygame.display.set_mode((BoardWidth,BoardHeight))
pygame.display.set_caption(title)

FPS = 10
fpsClock = pygame.time.Clock()

def printThreshold(thr):
    print("! Changed threshold to " + str(thr))

def Game():

    #参数设置
    bgModel = None
    cap_region_x_begin = 0.5  # start point/total width
    cap_region_y_end = 0.8  # start point/total width   
    threshold = 60  # BINARY threshold（making picture obvious)   
    blurValue = 41  # GaussianBlur parameter(smoothing picture)   
    bgSubThreshold = 50    
    learningRate = 0
    
    # variables    
    isBgCaptured = 0  # bool, whether the background captured    
    triggerSwitch = False  # if true, keyborad simulator works
    print("press 'b' to capture your background.")
    print("press 'n' to capture your gesture.")
        
    # Camera
    camera = cv2.VideoCapture(0)  
    camera.set(10, 200)
    cv2.namedWindow('trackbar')   
    cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)   
    while camera.isOpened():  # capture and convert image
    
        ret, frame = camera.read() 
        threshold = cv2.getTrackbarPos('trh1', 'trackbar')   
        frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter   
        frame = cv2.flip(frame, 1)  # flip the frame horizontally   
        cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),   
                      (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)    
        cv2.imshow('original', frame)
    
        #  Main operation  
        if isBgCaptured == 1:  #  background  is captured
    
            fgmask = bgModel.apply(frame, learningRate=learningRate)   
            kernel = np.ones((3, 3), np.uint8)      
            fgmask = cv2.erode(fgmask, kernel, iterations=1)      
            img = cv2.bitwise_and(frame, frame, mask=fgmask)  
            img = img[0:int(cap_region_y_end * frame.shape[0]),  
                  int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
            cv2.imshow('mask', img)
    
            # convert the image into binary image
    
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
            blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
            cv2.imshow('blur', blur) 
            ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY) 
            cv2.imshow('ori', thresh)
    
            # get the coutours
    
            thresh1 = copy.deepcopy(thresh)  
            contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
            length = len(contours)   
            maxArea = -1
    
            if length > 0:    
                for i in range(length):  # find the biggest contour (according to area)  
                    temp = contours[i]  
                    area = cv2.contourArea(temp)
    
                    if area > maxArea:
                        maxArea = area  
                        ci = i
    
                res = contours[ci]    
                hull = cv2.convexHull(res)    
                drawing = np.zeros(img.shape, np.uint8)    
                cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)    
                cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)    
                hull = cv2.convexHull(res, returnPoints=False)  # return the point index in the contour
                Flag = True
                
                if len(hull) > 3:
            
                    defects = cv2.convexityDefects(res, hull)   # finding defects            
                    if type(defects) != type(None):  # avoid crashing.   (BUG not found)           
                        cnt = 0           
                        for i in range(defects.shape[0]):  # calculate the angle
            
                            s, e, f, d = defects[i][0]            
                            start = tuple(res[s][0])            
                            end = tuple(res[e][0])            
                            far = tuple(res[f][0])            
                            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)            
                            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)            
                            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)            
                            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
            
                            if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers            
                                cnt += 1           
                                cv2.circle(drawing, far, 8, [211, 84, 0], -1)           
                        isFinishCal, cnt, Flag = True, cnt, False
                
                if (Flag!=False):
                    isFinishCal, cnt = False, 0
                              
                if triggerSwitch is True:
    
                    if isFinishCal is True and cnt <= 5:
                        #To determine what the player gesture represents
                        if cnt == 0:
                            print("stone")
                            camera.release()
                            cv2.destroyAllWindows()
                            break
                        elif cnt == 1:
                            print("scissors")
                            camera.release()
                            cv2.destroyAllWindows()
                            break
                        elif cnt == 4:
                            #Change the value of cnt for easy sorting later
                            cnt = 2
                            print("paper")
                            camera.release()
                            cv2.destroyAllWindows()
                            break
                        
            cv2.imshow('output', drawing)       # drawing the contour of one's gesture
    
        # Keyboard OP
    
        k = cv2.waitKey(10)    
        if k == 27:  # press ESC to exit    
            camera.release()    
            cv2.destroyAllWindows()
            break
    
        elif k == ord('b'):  # press 'b' to capture the background    
            bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)    
            isBgCaptured = 1   
            print('!!!Background Captured!!!')
    
        elif k == ord('r'):  # press 'r' to reset the background
    
            bgModel = None    
            triggerSwitch = False    
            isBgCaptured = 0    
            print('!!!Reset BackGround!!!')
    
        elif k == ord('n'):  # press 'n' to count the number 
    
            triggerSwitch = True  
            print('!!!Trigger On!!!')
            
    play = []
    play.append('rock')
    play.append('scissors')
    play.append('paper')
    p1 = cnt
    pc = random.randint(0, 2)
    print("you are ", play[p1], ",and the computer is ", play[pc], "\n")
    if (p1 == pc):
        print("Game Draw\n")
        game_result = 1
    if ((p1 == 0 and pc == 1) or (p1 == 1 and pc == 2) or (p1 == 2 and pc == 0)):
        print('you win!\n')
        game_result = 1
    else:
        print('you lose!\n')
        game_result = -1
    return game_result


# 插入文字，string表示需要插入的英文，cx、cy表示插入的坐标，fontSize表示字体大小，color表示字体颜色
def renderText(string,cx,cy,fontSize,color):
    fontObj = pygame.font.Font('freesansbold.ttf',fontSize)
    textSurfObj = fontObj.render(string,True,color)
    textRectObj = textSurfObj.get_rect()
    textRectObj.center = (cx,cy)
    dispsurf.blit(textSurfObj,textRectObj)

# 绘制摇骰子的按钮
def showButtonRoll():
    pygame.draw.rect(dispsurf,BLACK,(750,500,200,40))
    renderText("Roll",850,520,32,WHITE)

# 绘制移动的按钮
def showButtonMove():
    pygame.draw.rect(dispsurf,BLACK,(750,600,200,40))
    renderText("Move",850,620,32,WHITE)

# 绘制骰子所在的方框
def showDiceBox(n):
    pygame.draw.rect(dispsurf,BLACK,(750,450,200,40),3)
    renderText(str(n),850,470,32,BLACK)

# 绘制当前行动玩家的方框
def showTurnBox(n):
    pygame.draw.rect(dispsurf,BLACK,(750,10,200,40))
    pygame.draw.rect(dispsurf,WHITE,(750,10,200,40),1)
    pygame.draw.line(dispsurf,WHITE,(870,10),(870,50),1)
    renderText("Turn",810,30,32,WHITE)
    renderText("P"+str(n),910,30,32,WHITE)

# 绘制棋子初始位置的边框
def showLudoButtons():
    pygame.draw.rect(dispsurf,BLACK,(750,550,40,40),3)
    renderText(str(1),770,570,32,BLACK)

    pygame.draw.rect(dispsurf,BLACK,(800,550,40,40),3)
    renderText(str(2),820,570,32,BLACK)

    pygame.draw.rect(dispsurf,BLACK,(850,550,40,40),3)
    renderText(str(3),870,570,32,BLACK)

    pygame.draw.rect(dispsurf,BLACK,(900,550,40,40),3)
    renderText(str(4),920,570,32,BLACK)

# 绘制棋盘
def drawBoard():
    # 绘制外边框
    pygame.draw.line(dispsurf,BLACK,(0,0),(1000,0))
    pygame.draw.line(dispsurf,BLACK,(701,0),(701,700))
    
    # 绘制终点区域
    C = (350,350)
    TL = (275,275)
    BL = (275,425)
    BR = (425,425)
    TR = (425,275)
    pygame.draw.polygon(dispsurf,RED,(C,BR,BL),0)
    pygame.draw.polygon(dispsurf,GREEN,(C,BL,TL),0)
    pygame.draw.polygon(dispsurf,YELLOW,(C,TL,TR),0)
    pygame.draw.polygon(dispsurf,BLUE,(C,TR,BR),0)

    # 绘制上下左右四个轨道部分
    # 下方轨道
    list1 = [(275, 425,GREY), (275, 481,GREY), (275, 537,GREY), (275, 593,GREY), (275, 649,GREY),
(325, 425,RED), (325, 481,RED), (325, 537,RED), (325, 593,RED), (325, 649,GREY),
(375, 425,GREY), (375, 481,GREY), (375, 537,GREY), (375, 593,GREY), (375, 649,GREY)]
    for item in list1:
        if(item[2] == RED):
            pygame.draw.rect(dispsurf,item[2],(item[0],item[1],49,54),0)
        elif(item[2] == GREY):
            pygame.draw.rect(dispsurf,item[2],(item[0],item[1],49,54),0)
        else:
            pygame.draw.rect(dispsurf,BLACK,(item[0],item[1],50,54),1)

    # 左方轨道
    list2 = [(275, 275,GREY), (219, 275,GREY), (163, 275,GREY), (107, 275,GREY), (51, 275,GREY),
(275, 325,GREEN), (219, 325,GREEN), (163, 325,GREEN), (107, 325,GREEN), (51, 325,GREY),
(275, 375,GREY), (219, 375,GREY), (163, 375,GREY), (107, 375,GREY), (51, 375,GREY)]
    for item in list2:
        if(item[2] == GREEN):
            pygame.draw.rect(dispsurf,item[2],(item[0],item[1],-52,48),0)
        elif(item[2] == GREY):
            pygame.draw.rect(dispsurf,item[2],(item[0],item[1],-52,48),0)
        else:
            pygame.draw.rect(dispsurf,BLACK,(item[0],item[1],-52,48),1)

    # 上方轨道
    list3 = [(425, 275,GREY), (425, 219,GREY), (425, 163,GREY), (425, 107,GREY), (425, 51,GREY),
(375, 275,YELLOW), (375, 219,YELLOW), (375, 163,YELLOW), (375, 107,YELLOW), (375, 51,GREY),
(325, 275,GREY), (325, 219,GREY), (325, 163,GREY), (325, 107,GREY), (325, 51,GREY)]
    for item in list3:
        if(item[2] == YELLOW):
            pygame.draw.rect(dispsurf,item[2],(item[0],item[1],-46,-52),0)
        if(item[2] == GREY):
            pygame.draw.rect(dispsurf,item[2],(item[0],item[1],-46,-52),0)
        else:
            pygame.draw.rect(dispsurf,BLACK,(item[0],item[1],-46,-52),1)


    # 右方轨道
    list4 = [(425, 425,GREY), (481, 425,GREY), (537, 425,GREY), (593, 425,GREY), (649, 425,GREY),
(425, 375,BLUE), (481, 375,BLUE), (537, 375,BLUE), (593, 375,BLUE), (649,375,GREY),
(425, 325,GREY), (481, 325,GREY), (537, 325,GREY), (593, 325,GREY), (649, 325,GREY)]
    for item in list4:
        if(item[2] == BLUE):
            pygame.draw.rect(dispsurf,item[2],(item[0],item[1],54,-46),0)
        if(item[2] == GREY):
            pygame.draw.rect(dispsurf,item[2],(item[0],item[1],54,-46),0)
        else:
            pygame.draw.rect(dispsurf,BLACK,(item[0],item[1],54,-46),1)


    # 绘制起点轨道
    colors = [RED,GREEN,YELLOW,BLUE]
    p_home = [(0,425),(0,0),(425,0),(425,425)]

    for i in range(1,5):
        pygame.draw.rect(dispsurf,colors[i-1],(p_home[i-1][0]+1,p_home[i-1][1]+1,275,275))
        pygame.draw.rect(dispsurf,WHITE,(p_home[i-1][0]+41,p_home[i-1][1]+41,193,193))
        xx = p_home[i-1][0]+41
        yy = p_home[i-1][1]+41
        pygame.draw.rect(dispsurf,colors[i-1],(xx+15,yy+15,70,70))
        pygame.draw.rect(dispsurf,colors[i-1],(xx+105,yy+15,70,70))
        pygame.draw.rect(dispsurf,colors[i-1],(xx+15,yy+105,70,70))
        pygame.draw.rect(dispsurf,colors[i-1],(xx+105,yy+105,70,70))

    # 绘制起点箭头
    # 左上方
    pygame.draw.rect(dispsurf,(50,200,50),(73,235,17,20))
    pygame.draw.polygon(dispsurf,(50,200,50),((63,255),(100,255),(81.5,275)),0)
    # 左下方
    pygame.draw.rect(dispsurf,(200,50,50),(235,607,20,17))
    pygame.draw.polygon(dispsurf,(200,50,50),((255,597),(255,634),(275,615.5)),0)
    # 右上方
    pygame.draw.rect(dispsurf,(230,230,50),(445,71,20,17))
    pygame.draw.polygon(dispsurf,(230,230,50),((445,61),(445,97),(425,79.5)),0)
    # 右下方
    pygame.draw.rect(dispsurf,(87,150,255),(610,445,17,20))
    pygame.draw.polygon(dispsurf,(87,150,255),((600,445),(637,445),(618.5,425)),0)

    # 绘制区域标号
    renderText("Player 2",140,137,20,BLACK)
    renderText("Player 1",140,562,20,BLACK)
    renderText("Player 3",560,137,20,BLACK)
    renderText("Player 4",560,562,20,BLACK)
    

# 展示得分版
def displayScoreBoard(scores):
    pygame.draw.rect(dispsurf,BLACK,(750,50,200,40))
    renderText("Scores",(850,70),32,WHITE)

    pygame.draw.rect(dispsurf, BLACK, (750, 90, 200, 60))
    renderText("1", (775, 100), 20, WHITE)
    renderText("2", (825, 100), 20, WHITE)
    renderText("3", (875, 100), 20, WHITE)
    renderText("4", (925, 100), 20, WHITE)

    pygame.draw.line(dispsurf, WHITE, (750, 110), (950, 110), 1)
    pygame.draw.line(dispsurf, WHITE, (800, 110), (800, 150), 1)
    pygame.draw.line(dispsurf, WHITE, (850, 110), (850, 150), 1)
    pygame.draw.line(dispsurf, WHITE, (900, 110), (900, 150), 1)
    pygame.draw.line(dispsurf, WHITE, (950, 110), (950, 150), 1)

    renderText(str(scores[1]), (775, 130), 28, WHITE)
    renderText(str(scores[1]), (825, 130), 28, WHITE)
    renderText(str(scores[1]), (875, 130), 28, WHITE)
    renderText(str(scores[1]), (925, 130), 28, WHITE)

# 记录游戏得分
wo = 1
winOrder = [0,0,0,0]
def winingOrder(scores):
    global wo
    global winOrder
    for i in range(4):
        if(winOrder[i] == 0 and scores [i] == 1):
            winOrder[i] = wo
            wo += 1
    y = 170
    for i in range(4):
        if(winOrder[i] != 0):
            y+=50


# 棋子类
class LudoButton:
    radius = 15
    center_x = 0
    center_y = 0

    #绘制一个棋子
    def drawLudoButton(self,color,center_x,center_y,number):
        self.center_x = center_x
        self.center_y = center_y
        pygame.draw.circle(dispsurf,color,(self.center_x,self.center_y),self.radius,0)
        renderText(str(number), self.center_x,self.center_y, 20, BLACK)
        ##这个地方加上棋子的编号


# 玩家类
class Player (LudoButton):
    score = 0    # 记录每个玩家的得分
    def __init__(self,color,track):
        self.color = color    # 该玩家棋子的颜色
        self.LudobuttonList = [LudoButton() for i in range(1,5)]   # 初始化四个棋子
        self.finalstatus = [0,0,0,0]                               # 棋子是否到达终点的标记
        self.initialstatus = [0,0,0,0]                             # 棋子是否离开待行区的标记
        self.track = track                                         # 玩家运动的轨道
        self.currentposition = [-1,-1,-1,-1] # 当前每个棋子的位置（用track列表的下标表示，-1代表在准备区

    def placeLudoButtons(self,position,initiated):
        '''
        将棋子放在各自的位置
        position棋子应当摆放的位置
        initiated表示棋子的初始位置
        '''
        for i in range(4):
            if(initiated[i] == position[i]):
                kkk=32
            else:
                pygame.draw.circle(dispsurf,PURPLE,position[i-1],17,0)
                self.drawLudoButton(self.color,position[i-1][0],position[i-1][1],i+1)

    def makeAMove(self,current):
        '''
        对棋子进行移动一步
        current代表这个棋子移动前的位置
        '''
        pygame.draw.circle(dispsurf,self.color,self.track[current + 1],self.radius,0)
        
    def backToStart(self):
        '''
        对棋子进行移动一步
        current代表这个棋子移动前的位置
        '''
        pygame.draw.circle(dispsurf,self.color,self.track[0],self.radius,0)
        
    def backToEnd(self):
        '''
        将棋子移动到终点轨道
        '''
        pygame.draw.circle(dispsurf,self.color,self.track[41],self.radius,0)

    def InitiateStart(self, btn_no):
        '''
        将棋子放在起始位置
        btn_no指定进行初始化的棋子编号
        '''
        pygame.draw.circle(dispsurf,self.color,self.track[0],self.radius,0)


    def isInHome(self):
        '''
        判断有多少个棋子到达终点
        同时将到达终点棋子的finalstatus赋值为1
        '''
        for i in range (4):
            if(self.currentposition[i] >= 47 and self.finalstatus[i] != 1):
                self.score +=1
                self.finalstatus[i] = 1


    def displayTransitButtons(self):
        """
        展示某个玩家当前的所有棋子
        """
        for i in range(4):
            if(self.currentposition[i] != -1):
                pygame.draw.circle(dispsurf,PURPLE,self.track[self.currentposition[i]],17,0)
                self.drawLudoButton(self.color,self.track[self.currentposition[i]][0],self.track[self.currentposition[i]][1],i+1)
    
    

'''
 玩家一（红色棋子）的运动轨道
'''
track1 = [(300,621),(300,565),(300,509),(300,453),(247,400),(191,400),(135,400),(79,400),(23,400), (23,350),
(23,300),(79,300),(135,300),(191,300),(247,300),(300,247),(300,191),(300,135),(300,79),(300,23),(350,23),
(400,23),(400,79),(400,135),(400,191),(400,247),(453,300),(509,300),(565,300),(621,300),(677,300),(677,350),
(677,400),(621,400),(565,400),(509,400),(453,400),(400,453),(400,509),(400,565),(400, 621),(400,677),(350,677),
(350, 621),(350,565),(350,509),(350,453),(350,350),(350,350),(350,350),(350,350),(350,350),(350,350)]

'''
 玩家二（绿色棋子）的运动轨道
'''

track2 = [(79,300),(135,300),(191,300),(247,300),(300,247),(300,191),(300,135),(300,79),(300,23),
(350,23),(400,23),(400,79),(400,135),(400,191),(400,247),(453,300),(509,300),(565,300),(621,300),
(677,300),(677,350),(677,400),(621,400),(565,400),(509,400),(453,400),(400,453),(400,509),(400,565),
(400, 621),(400,677),(350,677),(300,677),(300,621),(300,565),(300,509),(300,453),(247,400),(191,400),(135,400),
(79,400),(23,400), (23,350),(79,350),(135,350),(191,350),(247,350),(350,350),(350,350),(350,350),(350,350),
(350,350),(350,350)]

'''
 玩家三（黄色棋子）的运动轨道
'''

track3 = [(400,79),(400,135),(400,191),(400,247),(453,300),(509,300),(565,300),(621,300),(677,300),
(677,350),(677,400),(621,400),(565,400),(509,400),(453,400),(400,453),(400,509),(400,565),(400, 621),
(400,677),(350,677),(300,677),(300,621),(300,565),(300,509),(300,453),(247,400),(191,400),(135,400),
(79,400),(23,400),(23,350),(23,300),(79,300),(135,300),(191,300),(247,300),(300,247),(300,191),(300,135),
(300,79),(300,23),(350,23),(350,79),(350,135),(350,191),(350,247),(350,350),(350,350),(350,350),(350,350),
(350,350),(350,350)]

'''
 玩家四（蓝色棋子）的运动轨道
'''

track4 = [(621, 400), (565, 400), (509, 400), (453, 400), (400, 453), (400, 509), (400, 565), (400,  621),
(400, 677),(350, 677), (300,677),(300, 621), (300, 565), (300, 509), (300, 453), (247, 400), (191, 400),
(135, 400), (79, 400),(23, 400), (23, 350), (23, 300), (79, 300), (135, 300), (191, 300),
(247, 300), (300, 247),(300, 191), (300, 135), (300, 79), (300, 23), (350, 23), (400, 23), (400, 79),
(400, 135), (400, 191),(400, 247), (453, 300), (509, 300), (565, 300), (621, 300), (677, 300), (677, 350),
(621, 350), (565, 350),(509, 350), (453, 350),(350,350),(350,350),(350,350),(350,350),(350,350),(350,350)]


mousex=0              # 记录鼠标单击区域的x坐标
mousey=0              # 记录鼠标单击区域的y坐标
mouseclicked = False  # 记录当前鼠标是否发生了单击
mc = 0                # 记录当前骰子是否要改变状态
movebutton = 0        # 记录哪一个棋子（1，2，3，4）要被移动
p_id = 1              # 当前正在行动的玩家
i=1                   # 记录棋子已经的运动步数
n=0                   # 记录摇骰子得到的点数
flagIMP = 0           # 记录是否完成了行走
bomb_place = []       # 记录炸弹的位置
star_place = []       # 记录星星的位置

# 显示骰子投掷的过程
def diceroll(n,c,b,tick,In):
    index = 1;
    location = (710, 140)  #骰子投掷的位置坐标
    
    #开始展示投掷骰子的动画
    while index < 15 :
        # In表示展示的图片编号，n代表最终骰子摇到的点数，b用于判断是否摇骰子
        if In < 15 and n == 1 and b == 1:
            filename = "./images/" + str(In) + ".gif"  
            img = pygame.image.load(filename)
            dispsurf.blit(img, location)
            c.tick(tick)
            In += 1
            #摇动14次之后，显示最终的点数
            if In == 14:
                filename = "./images/img1.gif"  
                img = pygame.image.load(filename)
                dispsurf.blit(img, location)
                c.tick(tick)
                b = 0
                In = 1
                
        if In < 15 and n == 2 and b == 1:
            filename = "./images/" + str(In) + ".gif"  
            img = pygame.image.load(filename)
            dispsurf.blit(img, location)
            c.tick(tick)
            In += 1
            if In == 14:
                filename = "./images/img2.gif"  
                img = pygame.image.load(filename)
                dispsurf.blit(img, location)
                c.tick(tick)
                b = 0
                In = 1
                
        if In < 15 and n == 3 and b == 1:
            filename = "./images/" + str(In) + ".gif"  
            img = pygame.image.load(filename)
            dispsurf.blit(img, location)
            c.tick(tick)
            In += 1
            if In == 14:
                filename = "./images/img3.gif"  
                img = pygame.image.load(filename)
                dispsurf.blit(img, location)
                c.tick(tick)
                b = 0
                In = 1
                
        if In < 15 and n == 4 and b == 1:
            filename = "./images/" + str(In) + ".gif"  
            img = pygame.image.load(filename)
            dispsurf.blit(img, location)
            c.tick(tick)
            In += 1
            if In == 14:
                filename = "./images/img4.gif"  
                img = pygame.image.load(filename)
                dispsurf.blit(img, location)
                c.tick(tick)
                b = 0
                In = 1
                
        if In < 15 and n == 5 and b == 1:
            filename = "./images/" + str(In) + ".gif"  
            img = pygame.image.load(filename)
            dispsurf.blit(img, location)
            c.tick(tick)
            In += 1
            if In == 14:
                filename = "./images/img5.gif"  
                img = pygame.image.load(filename)
                dispsurf.blit(img, location)
                c.tick(tick)
                b = 0
                In = 1
                
        if In < 15 and n == 6 and b == 1:
            filename = "./images/" + str(In) + ".gif"  
            img = pygame.image.load(filename)
            dispsurf.blit(img, location)
            c.tick(tick)
            In += 1
            if In == 14:
                filename = "./images/img6.gif"  
                img = pygame.image.load(filename)
                dispsurf.blit(img, location)
                c.tick(tick)
                b = 0
                In = 1
                
        pygame.display.flip()
        index += 1

# 显示骰子投掷的结果
def dispdiceroll(n):
    location = (710,140)
    if n == 1:
        filename = "./images/img1.gif"  
        img = pygame.image.load(filename)
        dispsurf.blit(img, location)
    elif n == 2:
        filename = "./images/img2.gif" 
        img = pygame.image.load(filename)
        dispsurf.blit(img, location)
    elif n == 3:
        filename = "./images/img3.gif" 
        img = pygame.image.load(filename)
        dispsurf.blit(img, location)
    elif n == 4:
        filename = "./images/img4.gif" 
        img = pygame.image.load(filename)
        dispsurf.blit(img, location)
    elif n == 5:
        filename = "./images/img5.gif"  
        img = pygame.image.load(filename)
        dispsurf.blit(img, location)
    elif n == 6:
        filename = "./images/img6.gif" 
        img = pygame.image.load(filename)
        dispsurf.blit(img, location)

# 展示炸弹的图片
def displaybomb(location):
    filename = "./images/Bomb1.png"  
    img = pygame.image.load(filename)
    img = pygame.transform.scale(img,(41,41))
    x,y = location
    x = x-20
    y = y-22
    dispsurf.blit(img, (x,y))

# 展示星星的图片
def displaystar(location):
    filename = "./images/star.png"  
    img = pygame.image.load(filename)
    img = pygame.transform.scale(img,(41,41))
    x,y = location
    x = x-20
    y = y-22
    dispsurf.blit(img, (x,y))
    
'''
  主函数
'''

c = pygame.time.Clock() 
b=0
tick = 20
In=1


        
'''
 构建四个玩家的对象
'''
Player1 = Player(RED, track1)
Player2 = Player(GREEN, track2)
Player3 = Player(YELLOW, track3)
Player4 = Player(BLUE, track4)

'''
 四个玩家棋子的初始位置列表
'''
position1 = [(91,516), (181,516), (91,606), (181,606)]
position2 = [(91,91), (181,91), (91,181), (181,181)]
position3 = [(516,91), (606,91), (516,181), (606,181)]
position4 = [(516,516), (606,516), (516,606), (606,606)]


'''
 四个玩家初始的棋子状态l列表
'''
initiated1 = [(0,0),(0,0),(0,0),(0,0)]
initiated2 = [(0,0),(0,0),(0,0),(0,0)]
initiated3 = [(0,0),(0,0),(0,0),(0,0)]
initiated4 = [(0,0),(0,0),(0,0),(0,0)]

while True:
    dispsurf.fill(WHITE)    # 背景颜色设置为白色
    drawBoard()             # 绘制画布
    showButtonRoll()        # 绘制摇骰子按钮
    showButtonMove()        # 绘制移动按钮
    dispdiceroll(n)         # 显示骰子的状态

    if(mc == 0):            # 当摇骰子的时候保持之前所显示的数字
        showDiceBox(0)
    else:
        showDiceBox(n)      # 摇完骰子后，显示摇到的数字

    showLudoButtons()       # 显示用于控制第几个棋子行走的按钮
    showTurnBox(p_id)       # 显示当前正在激活的玩家


    # 展示炸弹位置
    for Item in bomb_place:
        displaybomb(Item)
    
    # 展示星星位置
    for Item in star_place:
        displaystar(Item)
        
    # 将四个玩家的棋子摆放在合适的位置

    Player1.placeLudoButtons(position1,initiated1)         # 玩家 1
    Player2.placeLudoButtons(position2,initiated2)         # 玩家 2
    Player3.placeLudoButtons(position3,initiated3)         # 玩家 3
    Player4.placeLudoButtons(position4,initiated4)         # 玩家 4

    # 展示棋子

    Player1.displayTransitButtons()                        # 玩家 1
    Player2.displayTransitButtons()                        # 玩家 2
    Player3.displayTransitButtons()                        # 玩家 3
    Player4.displayTransitButtons()                        # 玩家 4
    
        
    # 记录四个人的成绩列表
    scores = [Player1.score,Player2.score,Player3.score,Player4.score]
  

    for event in pygame.event.get():  # 获取当前事件

        if event.type == QUIT:        # 退出游戏
            pygame.quit()
            sys.exit()
        elif event.type == MOUSEBUTTONUP:    # 鼠标单击某处
            (mousex,mousey) = event.pos      # 记录鼠标点击处的坐标
            mouseclicked = True              # 记录当前鼠标发生了单击


    #事件1：roll按钮被单击
    if(mouseclicked == True and mousex >= 750 and mousex <= 950 and mousey >= 500 and mousey <= 540):
        mc = 1
        n = randint(1,6)       # 获取一个0-6之间的随机数
        b=1
        mouseclicked = False   # 恢复鼠标初始的状态
        
        ''' 开始摇骰子'''
        diceroll(n, c, b, tick, In)
        ''' 结束摇骰子'''
        
        showDiceBox(n)         # 显示骰子上的数字



    #事件2：单击选择某个棋子
    elif(mouseclicked == True and mousey >= 550 and mousey <= 590):  
        if(mousex >= 750 and mousex <= 790):  # 第一个棋子被选中                     
            movebutton = 1
            
        elif(mousex >= 800 and mousex <= 840): # 第二个棋子被选中
            movebutton = 2
            
        elif(mousex >= 850 and mousex <= 890):  # 第三个棋子被选中
            movebutton = 3   
                           
        elif(mousex >= 900 and mousex <= 940):  # 第四个棋子被选中
            movebutton = 4  
                          
        mouseclicked = False             # 恢复鼠标初始的状态

    # 事件3：单击move按钮
    elif(mouseclicked == True and mousex >= 750 and mousex <= 950 and mousey >= 600 and mousey <= 640):
        
        if(p_id == 1):      # 当前玩家1正在行动
            
            #如果所有的棋子都到达终点，则改变当前正在行动的玩家
            if(Player1.finalstatus[0] == 1 and Player1.finalstatus[1] == 1 and  Player1.finalstatus[2] == 1 and Player1.finalstatus[3] == 1):
                p_id = 2    
                
            #如果有棋子没有到达终点    
            else:
                if(movebutton ==1):      # 第一个棋子被选择
                    
                    #如果刚刚行动完毕，计算当前棋子的位置距离终点的距离
                    if(flagIMP == 0):
                        diff = (47 - Player1.currentposition[0])
                        flagIMP = 1

                    #如果当前这个棋子在待行区
                    if(Player1.initialstatus[0] == 0):  
                        if(n == 6 or n == 1):     # 如果骰子摇到1或者6，可以初始化棋子

                            Player1.initialstatus[0] = 1    # 记录玩家1的棋子1的初始化状态
                            initiated1[0] = position1[0]    # 将棋子放在起始的位置
                            Player1.InitiateStart(1)        # 对玩家1的棋子1进行初始化
                            mouseclicked = False            # 恢复鼠标初始的状态
                            Player1.currentposition[0] = 0  # 设定棋子1当前的位置

                        else:
                            p_id = 2                        # 如果不是1或者6，则改变当前正在行动的玩家
                            mouseclicked =False             # 恢复鼠标初始的状态
                    
                    #如果这个棋子不在待行区
                    else:

                        if(i<=n):  #始终移动直到移动总步数i大于n为止
                            
                            #结束了该次执骰子的最后一步，进行记录
                            if(i==n):
                                flagIMP = 0
 
                            #如果这个棋子没有到达终点
                            if(Player1.finalstatus[0] !=1):
                                
                                #如果当前棋子已经进入终点轨道，需要判断该次行走是否会导致超过终点
                                if(Player1.currentposition[0] >= 41):
                                    Player1.makeAMove(Player1.currentposition[0])#将该棋子向前移动一步
                                    Player1.currentposition[0] +=1  #记录棋子当前的位置
                                    i+=1 #记录当前已经向前移动的步数
                                    if track1[Player1.currentposition[0]] == (350,350):
                                        i = n+1
                                    
                                #如果当前棋子没有进入终点轨道，则正常前进
                                elif(Player1.currentposition[0] < 41):
                                    Player1.makeAMove(Player1.currentposition[0])#将该棋子向前移动一步
                                    Player1.currentposition[0] +=1#记录棋子当前的位置
                                    i+=1 #记录当前已经向前移动的步数
                                
                                #如果这次行走会使得棋子超过终点，则不行动
                                else:
                                    p_id = 2  #重置当前行动的棋子
                                    i=1  #重置已经向前的步数
                                    mouseclicked = False   # 恢复鼠标初始的状态

                            else:
                                renderText("This chess has been to the destination!",850,70,15,BLACK)
                                renderText("Please change another button!.",850,85,15,BLACK)
                        
                        #如果前进步数已经为n步，则执行相应的触发行动
                        elif(i > n):
                            # 如果到达的位置有炸弹，则需要玩游戏，输的时候需要将棋子退回到初始位置
                            flag = False
                            for bombPosition in bomb_place:
                                # 当前到达的位置有炸弹
                                if track1[Player1.currentposition[0]] == bombPosition:
                                    flag = True
                                    bomb_place.remove(bombPosition)  # 删除当前位置的炸弹
                                    game_result = Game()                    # 进行一次剪刀石头布游戏
                                    
                                    # 输掉游戏
                                    if game_result == -1:
                                        Player1.currentposition[0] = 0   # 棋子的位置设定为初始的位置
                                        Player1.backToStart()            # 将棋子放置到起点的位置
                            
                            for starPosition in star_place:
                                # 当前到达的位置有星星
                                if track1[Player1.currentposition[0]] == starPosition:
                                    flag = True
                                    star_place.remove(starPosition)  # 删除当前位置的星星

                                    Player1.currentposition[0] = 41   # 棋子的位置设定为终点轨道
                                    Player1.backToEnd()            # 将棋子放置到终点轨道位置  
                                        
                            # 如果投掷到的骰子为6，则可以安放一个炸弹
                            if (n==6) and (flag == False) and (track1[Player1.currentposition[0]] != (350,350)) and (len(bomb_place)<5):
                                bomb_place.append(track1[Player1.currentposition[0]])
                            
                            # 如果投掷到的骰子为1，则可以安放一个星星
                            if (n==1) and (flag == False) and (track1[Player1.currentposition[0]] != (350,350)) and (len(star_place)<5):
                                star_place.append(track1[Player1.currentposition[0]])

                            p_id = 2 #重置当前行动的棋子
                            i=1 #重置已经向前的步数
                            mouseclicked = False # 恢复鼠标初始的状态
                            
                #第二个棋子被选择，算法逻辑同上
                if(movebutton ==2):
                    if(flagIMP == 0):
                        diff = (47 - Player1.currentposition[1])
                        flagIMP = 1

                    if(Player1.initialstatus[1] == 0):
                        if(n == 6 or n == 1):

                            Player1.initialstatus[1] = 1
                            initiated1[1] = position1[1]
                            Player1.InitiateStart(2)
                            mouseclicked = False
                            Player1.currentposition[1] = 0
                            
                        else:
                            p_id = 2
                            mouseclicked =False
                    else:

                        if(i<=n):
                            if(i==n):
                                flagIMP = 0

                            if(Player1.finalstatus[1] !=1):
                                if(Player1.currentposition[1] >= 41):
                                    Player1.makeAMove(Player1.currentposition[1])
                                    Player1.currentposition[1] +=1
                                    i+=1
                                    if track1[Player1.currentposition[1]] == (350,350):
                                        i = n+1
                                        
                                elif(Player1.currentposition[1] < 41):
                                    Player1.makeAMove(Player1.currentposition[1])
                                    Player1.currentposition[1] +=1
                                    i+=1
                                else:
                                    p_id = 2
                                    i=1
                                    mouseclicked = False


                            else:
                                renderText("This chess has been to the destination!",850,70,15,BLACK)
                                renderText("Please change another button!.",850,85,15,BLACK)

                        elif(i > n):
                            
                            flag = False
                            for bombPosition in bomb_place:

                                if track1[Player1.currentposition[1]] == bombPosition:
                                    flag = True
                                    bomb_place.remove(bombPosition)  
                                    game_result = Game()                  
                                    
                                    if game_result == -1:
                                        Player1.currentposition[1] = 0   
                                        Player1.backToStart()       
                                        
                            for starPosition in star_place:
             
                                if track1[Player1.currentposition[1]] == starPosition:
                                    flag = True
                                    star_place.remove(starPosition)  

                                    Player1.currentposition[1] = 41   
                                    Player1.backToEnd()            
                                    
                            if (n==6) and (flag == False) and (track1[Player1.currentposition[1]] != (350,350)) and (len(bomb_place)<5):
                                bomb_place.append(track1[Player1.currentposition[1]])
                            
                            if (n==1) and (flag == False) and (track1[Player1.currentposition[1]] != (350,350)) and (len(star_place)<5):
                                star_place.append(track1[Player1.currentposition[1]])
                                
                            p_id = 2
                            i=1
                            mouseclicked = False
                
                #第三个棋子被选择，算法逻辑同上
                if(movebutton ==3):
                    if(flagIMP == 0):
                        diff = (47 - Player1.currentposition[2])
                        flagIMP = 1

                    if(Player1.initialstatus[2] == 0):
                        
                        if(n == 6 or n == 1):
                            Player1.initialstatus[2] = 1
                            initiated1[2] = position1[2]
                            Player1.InitiateStart(3)
                            mouseclicked = False
                            Player1.currentposition[2] = 0

                        else:
                            p_id = 2
                            mouseclicked =False
                    else:

                        if(i<=n):
                            if(i==n):
                                flagIMP = 0

                            if(Player1.finalstatus[2] !=1):
                                
                                if(Player1.currentposition[2] >= 41):
                                    Player1.makeAMove(Player1.currentposition[2])
                                    Player1.currentposition[2] +=1
                                    i+=1
                                    if track1[Player1.currentposition[2]] == (350,350):
                                        i = n+1

                                elif(Player1.currentposition[2] < 41):
                                    Player1.makeAMove(Player1.currentposition[2])
                                    Player1.currentposition[2] +=1
                                    i+=1
                                else:
                                    p_id = 2
                                    i=1
                                    mouseclicked = False

                            else:
                                renderText("This chess has been to the destination!",850,70,15,BLACK)
                                renderText("Please change another button!.",850,85,15,BLACK)

                        elif(i > n):
                            
                            flag = False
                            for bombPosition in bomb_place:

                                if track1[Player1.currentposition[2]] == bombPosition:
                                    flag = True
                                    bomb_place.remove(bombPosition)  
                                    game_result = Game()                  
                                    
                                    if game_result == -1:
                                        Player1.currentposition[2] = 0   
                                        Player1.backToStart()           
                            
                            for starPosition in star_place:
             
                                if track1[Player1.currentposition[2]] == starPosition:
                                    flag = True
                                    star_place.remove(starPosition)  

                                    Player1.currentposition[2] = 41   
                                    Player1.backToEnd()    
                                    
                            if (n==6) and (flag == False) and (track1[Player1.currentposition[2]] != (350,350)) and (len(bomb_place)<5):
                                bomb_place.append(track1[Player1.currentposition[2]])
                            
                            if (n==1) and (flag == False) and (track1[Player1.currentposition[2]] != (350,350)) and (len(star_place)<5):
                                star_place.append(track1[Player1.currentposition[2]])
                                
                            p_id = 2
                            i=1
                            mouseclicked = False
                            
                            
                if(movebutton ==4):
                    if(flagIMP == 0):
                        diff = (47 - Player1.currentposition[3])
                        flagIMP = 1

                    if(Player1.initialstatus[3] == 0):
                        if(n == 6 or n == 1):

                            Player1.initialstatus[3] = 1
                            initiated1[3] = position1[3]
                            Player1.InitiateStart(4)
                            mouseclicked = False
                            Player1.currentposition[3] = 0
                            
                        else:
                            p_id = 2
                            mouseclicked =False
                    
                    else:
                        if(i<=n):
                            if(i==n):
                                flagIMP = 0

                            if(Player1.finalstatus[3] !=1):
                                if(Player1.currentposition[3] >= 41):
                                    Player1.makeAMove(Player1.currentposition[3])
                                    Player1.currentposition[3] +=1
                                    i+=1
                                    if track1[Player1.currentposition[3]] == (350,350):
                                        i = n+1
                                        
                                elif(Player1.currentposition[3] < 41):
                                    Player1.makeAMove(Player1.currentposition[3])
                                    Player1.currentposition[3] +=1
                                    i+=1
                                else:
                                    p_id = 2
                                    i=1
                                    mouseclicked = False

                            else:
                                renderText("This chess has been to the destination!",850,70,15,BLACK)
                                renderText("Please change another button!.",850,85,15,BLACK)

                        elif(i > n):
                            
                            flag = False
                            for bombPosition in bomb_place:

                                if track1[Player1.currentposition[3]] == bombPosition:
                                    flag = True
                                    bomb_place.remove(bombPosition)  
                                    game_result = Game()                  
                                    
                                    if game_result == -1:
                                        Player1.currentposition[3] = 0   
                                        Player1.backToStart()           
                            
                            for starPosition in star_place:
             
                                if track1[Player1.currentposition[3]] == starPosition:
                                    flag = True
                                    star_place.remove(starPosition)  

                                    Player1.currentposition[3] = 41   
                                    Player1.backToEnd()  
                                    
                            if (n==6) and (flag == False) and (track1[Player1.currentposition[3]] != (350,350)) and (len(bomb_place)<5):
                                bomb_place.append(track1[Player1.currentposition[3]])
                            
                            if (n==1) and (flag == False) and (track1[Player1.currentposition[3]] != (350,350)) and (len(star_place)<5):
                                star_place.append(track1[Player1.currentposition[3]])
                                
                            p_id = 2
                            i=1
                            mouseclicked = False

        elif(p_id == 2):   # 当前玩家2正在行动，逻辑同上
            
            
            if(Player2.finalstatus[0] == 1 and Player2.finalstatus[1] == 1 and Player2.finalstatus[2] == 1 and Player2.finalstatus[3] == 1):
                p_id = 3
                
            else:
                if(movebutton ==1):
                    if(flagIMP == 0):
                        diff = (47 - Player2.currentposition[0])
                        flagIMP = 1

                    if(Player2.initialstatus[0] == 0):
                        if(n == 6 or n == 1):
                            

                            Player2.initialstatus[0] = 1
                            initiated2[0] = position2[0]
                            Player2.InitiateStart(1)
                            mouseclicked = False
                            Player2.currentposition[0] = 0

                        else:
                            p_id = 3
                            mouseclicked =False
                    
                    else:
                        if(i<=n):
                            if(i==n):
                                flagIMP = 0

                            if(Player2.finalstatus[0] !=1):
                                if(Player2.currentposition[0] >= 41):
                                    Player2.makeAMove(Player2.currentposition[0])
                                    Player2.currentposition[0] +=1
                                    i+=1
                                    if track2[Player2.currentposition[0]] == (350,350):
                                        i = n+1
                                        
                                elif(Player2.currentposition[0] < 41):
                                    Player2.makeAMove(Player2.currentposition[0])
                                    Player2.currentposition[0] +=1
                                    i+=1
                                else:
                                    p_id = 3
                                    i=1
                                    mouseclicked = False

                            else:
                                renderText("This chess has been to the destination!",850,70,15,BLACK)
                                renderText("Please change another button!.",850,85,15,BLACK)

                        elif(i > n):
                            
                            flag = False
                            for bombPosition in bomb_place:

                                if track2[Player2.currentposition[0]] == bombPosition:
                                    flag = True
                                    bomb_place.remove(bombPosition)  
                                    game_result = Game()                  
                                    
                                    if game_result == -1:
                                        Player2.currentposition[0] = 0   
                                        Player2.backToStart()           
                            
                            for starPosition in star_place:
             
                                if track2[Player2.currentposition[0]] == starPosition:
                                    flag = True
                                    star_place.remove(starPosition)  

                                    Player2.currentposition[0] = 41   
                                    Player2.backToEnd()  
                                    
                            if (n==6) and (flag == False) and (track2[Player2.currentposition[0]] != (350,350)) and (len(bomb_place)<5):
                                bomb_place.append(track2[Player2.currentposition[0]])
                            
                            if (n==1) and (flag == False) and (track2[Player2.currentposition[0]] != (350,350)) and (len(star_place)<5):
                                star_place.append(track2[Player2.currentposition[0]])
                                
                            p_id = 3
                            i=1
                            mouseclicked = False
                
                if(movebutton ==2):

                    if(flagIMP == 0):
                        diff = (47 - Player2.currentposition[1])
                        flagIMP = 1

                    if(Player2.initialstatus[1] == 0):
                        if(n == 6 or n == 1):

                            Player2.initialstatus[1] = 1
                            initiated2[1] = position2[1]
                            Player2.InitiateStart(2)
                            mouseclicked = False
                            Player2.currentposition[1] = 0

                        else:
                            p_id = 3
                            mouseclicked =False
                    else:
                        if(i<=n):
                            if(i==n):
                                flagIMP = 0

                            if(Player2.finalstatus[1] !=1):
                                if(Player2.currentposition[1] >= 41):
                                    Player2.makeAMove(Player2.currentposition[1])
                                    Player2.currentposition[1] +=1
                                    i+=1
                                    if track2[Player2.currentposition[1]] == (350,350):
                                        i = n+1
                                        
                                elif(Player2.currentposition[1] < 41):
                                    Player2.makeAMove(Player2.currentposition[1])
                                    Player2.currentposition[1] +=1
                                    i+=1
                                else:
                                    p_id = 3
                                    i=1
                                    mouseclicked = False

                            else:
                                renderText("This chess has been to the destination!",850,70,15,BLACK)
                                renderText("Please change another button!.",850,85,15,BLACK)

                        elif(i > n):
                            
                            flag = False
                            for bombPosition in bomb_place:

                                if track2[Player2.currentposition[1]] == bombPosition:
                                    flag = True
                                    bomb_place.remove(bombPosition)  
                                    game_result = Game()                  
                                    
                                    if game_result == -1:
                                        Player2.currentposition[1] = 0   
                                        Player2.backToStart()           
                            
                            for starPosition in star_place:
             
                                if track2[Player2.currentposition[1]] == starPosition:
                                    flag = True
                                    star_place.remove(starPosition)  

                                    Player2.currentposition[1] = 41   
                                    Player2.backToEnd()  
                                    
                            if (n==6) and (flag == False) and (track2[Player2.currentposition[1]] != (350,350)) and (len(bomb_place)<5):
                                bomb_place.append(track2[Player2.currentposition[1]])
                            
                            if (n==1) and (flag == False) and (track2[Player2.currentposition[1]] != (350,350)) and (len(star_place)<5):
                                star_place.append(track2[Player2.currentposition[1]])
                                
                            p_id = 3
                            i=1
                            mouseclicked = False
                            
                if(movebutton ==3):
                    if(flagIMP == 0):
                        diff = (47 - Player2.currentposition[2])
                        flagIMP = 1

                    if(Player2.initialstatus[2] == 0):
                        if(n == 6 or n == 1):

                            Player2.initialstatus[2] = 1
                            initiated2[2] = position2[2]
                            Player2.InitiateStart(3)
                            mouseclicked = False
                            Player2.currentposition[2] = 0
                            
                        else:
                            p_id = 3
                            mouseclicked =False
                 
                    else:
                        if(i<=n):
                            if(i==n):
                                flagIMP = 0

                            if(Player2.finalstatus[2] !=1):

                                if(Player2.currentposition[2] >= 41):
                                    Player2.makeAMove(Player2.currentposition[2])
                                    Player2.currentposition[2] +=1
                                    i+=1
                                    if track2[Player2.currentposition[2]] == (350,350):
                                        i = n+1
                                        
                                elif(Player2.currentposition[2] < 41):
                                    Player2.makeAMove(Player2.currentposition[2])
                                    Player2.currentposition[2] +=1
                                    i+=1
                                else:
                                    p_id = 3
                                    i=1
                                    mouseclicked = False
                                    
                            else:
                                renderText("This chess has been to the destination!",850,70,15,BLACK)
                                renderText("Please change another button!.",850,85,15,BLACK)

                        elif(i > n):
                            
                            flag = False
                            for bombPosition in bomb_place:

                                if track2[Player2.currentposition[2]] == bombPosition:
                                    flag = True
                                    bomb_place.remove(bombPosition)  
                                    game_result = Game()                  
                                    
                                    if game_result == -1:
                                        Player2.currentposition[2] = 0   
                                        Player2.backToStart()           
                            
                            for starPosition in star_place:
             
                                if track2[Player2.currentposition[2]] == starPosition:
                                    flag = True
                                    star_place.remove(starPosition)  

                                    Player2.currentposition[2] = 41   
                                    Player2.backToEnd()  
                                    
                            if (n==6) and (flag == False) and (track2[Player2.currentposition[2]] != (350,350)) and (len(bomb_place)<5):
                                bomb_place.append(track2[Player2.currentposition[2]])
                            
                            if (n==1) and (flag == False) and (track2[Player2.currentposition[2]] != (350,350)) and (len(star_place)<5):
                                star_place.append(track2[Player2.currentposition[2]])
                                
                            p_id = 3
                            i=1
                            mouseclicked = False
                            
                if(movebutton ==4):
                    if(flagIMP == 0):
                        diff = (47 - Player2.currentposition[3])
                        flagIMP = 1

                    if(Player2.initialstatus[3] == 0):
                        if(n == 6 or n == 1):
                            Player2.initialstatus[3] = 1
                            initiated2[3] = position2[3]
                            Player2.InitiateStart(4)
                            mouseclicked = False
                            Player2.currentposition[3] = 0
                        else:
                            p_id = 1
                            mouseclicked =False
                
                    else:
                        if(i<=n):
                            if(i==n):
                                flagIMP = 0

                            if(Player2.finalstatus[3] !=1):
                                if(Player2.currentposition[3] >= 41):
                                    Player2.makeAMove(Player2.currentposition[3])
                                    Player2.currentposition[3] +=1
                                    i+=1
                                    if track2[Player2.currentposition[3]] == (350,350):
                                        i = n+1
                                        
                                elif(Player2.currentposition[3] < 41):
                                    Player2.makeAMove(Player2.currentposition[3])
                                    Player2.currentposition[3] +=1
                                    i+=1
                                else:
                                    p_id = 3
                                    i=1
                                    mouseclicked = False

                            else:
                                renderText("This chess has been to the destination!",850,70,15,BLACK)
                                renderText("Please change another button!.",850,85,15,BLACK)

                        elif(i > n):
                            
                            flag = False
                            for bombPosition in bomb_place:

                                if track2[Player2.currentposition[3]] == bombPosition:
                                    flag = True
                                    bomb_place.remove(bombPosition)  
                                    game_result = Game()                  
                                    
                                    if game_result == -1:
                                        Player2.currentposition[3] = 0   
                                        Player2.backToStart()           
                            
                            for starPosition in star_place:
             
                                if track2[Player2.currentposition[3]] == starPosition:
                                    flag = True
                                    star_place.remove(starPosition)  

                                    Player2.currentposition[3] = 41   
                                    Player2.backToEnd()  
                                    
                            if (n==6) and (flag == False) and (track2[Player2.currentposition[3]] != (350,350)) and (len(bomb_place)<5):
                                bomb_place.append(track2[Player2.currentposition[3]])
                            
                            if (n==1) and (flag == False) and (track2[Player2.currentposition[3]] != (350,350)) and (len(star_place)<5):
                                star_place.append(track2[Player2.currentposition[3]])
                                
                            p_id = 3
                            i=1
                            mouseclicked = False


        elif(p_id == 3):   # 当前玩家3正在行动，逻辑同上
            if(Player3.finalstatus[0] == 1 and Player3.finalstatus[1] == 1 and Player3.finalstatus[2] == 1 and Player3.finalstatus[3] == 1):
                p_id = 4
            else:
                if(movebutton ==1):
                    if(flagIMP == 0):
                        diff = (47 - Player3.currentposition[0])
                        flagIMP = 1

                    if(Player3.initialstatus[0] == 0):
                        if(n == 6 or n == 1):

                            Player3.initialstatus[0] = 1
                            initiated3[0] = position3[0]
                            Player3.InitiateStart(1)
                            mouseclicked = False
                            Player3.currentposition[0] = 0

                        else:
                            p_id = 4
                            mouseclicked =False
                    
                    else:
                        if(i<=n):
                            if(i==n):
                                flagIMP = 0

                            if(Player3.finalstatus[0] !=1):
                                if(Player3.currentposition[0] >= 41):
                                    Player3.makeAMove(Player3.currentposition[0])
                                    Player3.currentposition[0] +=1
                                    i+=1
                                    if track3[Player3.currentposition[0]] == (350,350):
                                        i = n+1
                                        
                                elif(Player3.currentposition[0] < 41):
                                    Player3.makeAMove(Player3.currentposition[0])
                                    Player3.currentposition[0] +=1
                                    i+=1
                                else:
                                    p_id = 4
                                    i=1
                                    mouseclicked = False

                            else:
                                renderText("This chess has been to the destination!",850,70,15,BLACK)
                                renderText("Please change another button!.",850,85,15,BLACK)

                        elif(i > n):
                            
                            flag = False
                            for bombPosition in bomb_place:

                                if track3[Player3.currentposition[0]] == bombPosition:
                                    flag = True
                                    bomb_place.remove(bombPosition)  
                                    game_result = Game()                  
                                    
                                    if game_result == -1:
                                        Player3.currentposition[0] = 0   
                                        Player3.backToStart()           
                            
                            for starPosition in star_place:
             
                                if track3[Player3.currentposition[0]] == starPosition:
                                    flag = True
                                    star_place.remove(starPosition)  

                                    Player3.currentposition[0] = 41   
                                    Player3.backToEnd()  
                                    
                            if (n==6) and (flag == False) and (track3[Player3.currentposition[0]] != (350,350)) and (len(bomb_place)<5):
                                bomb_place.append(track3[Player3.currentposition[0]])
                            
                            if (n==1) and (flag == False) and (track3[Player3.currentposition[0]] != (350,350)) and (len(star_place)<5):
                                star_place.append(track3[Player3.currentposition[0]])
                                
                            p_id = 4
                            i=1
                            mouseclicked = False
                            
                if(movebutton ==2):

                    if(flagIMP == 0):
                        diff = (47 - Player3.currentposition[1])
                        flagIMP = 1

                    if(Player3.initialstatus[1] == 0):
                        if(n == 6 or n == 1):

                            Player3.initialstatus[1] = 1
                            initiated3[1] = position3[1]
                            Player3.InitiateStart(2)
                            mouseclicked = False
                            Player3.currentposition[1] = 0

                        else:
                            p_id = 4
                            mouseclicked =False
                    
                    else:
                        if(i<=n):
                            if(i==n):
                                flagIMP = 0

                            if(Player3.finalstatus[1] !=1):
                                if(Player3.currentposition[1] >= 41):
                                    Player3.makeAMove(Player3.currentposition[1])
                                    Player3.currentposition[1] +=1
                                    i+=1
                                    if track3[Player3.currentposition[1]] == (350,350):
                                        i = n+1
                                        
                                elif(Player3.currentposition[1] < 41):
                                    Player3.makeAMove(Player3.currentposition[1])
                                    Player3.currentposition[1] +=1
                                    i+=1
                                else:
                                    p_id = 4
                                    i=1
                                    mouseclicked = False

                            else:
                                renderText("This chess has been to the destination!",850,70,15,BLACK)
                                renderText("Please change another button!.",850,85,15,BLACK)

                        elif(i > n):
                            
                            flag = False
                            for bombPosition in bomb_place:

                                if track3[Player3.currentposition[1]] == bombPosition:
                                    flag = True
                                    bomb_place.remove(bombPosition)  
                                    game_result = Game()                  
                                    
                                    if game_result == -1:
                                        Player3.currentposition[1] = 0   
                                        Player3.backToStart()           
                            
                            for starPosition in star_place:
             
                                if track3[Player3.currentposition[1]] == starPosition:
                                    flag = True
                                    star_place.remove(starPosition)  

                                    Player3.currentposition[1] = 41   
                                    Player3.backToEnd()  
                                    
                            if (n==6) and (flag == False) and (track3[Player3.currentposition[1]] != (350,350)) and (len(bomb_place)<5):
                                bomb_place.append(track3[Player3.currentposition[1]])
                            
                            if (n==1) and (flag == False) and (track3[Player3.currentposition[1]] != (350,350)) and (len(star_place)<5):
                                star_place.append(track3[Player3.currentposition[1]])
                                
                            p_id = 4
                            i=1
                            mouseclicked = False
                            
                if(movebutton ==3):
                    if(flagIMP == 0):
                        diff = (47 - Player3.currentposition[2])
                        flagIMP = 1

                    if(Player3.initialstatus[2] == 0):
                        if(n == 6 or n == 1):

                            Player3.initialstatus[2] = 1
                            initiated3[2] = position3[2]
                            Player3.InitiateStart(3)
                            mouseclicked = False
                            Player3.currentposition[2] = 0

                        else:
                            p_id = 4
                            mouseclicked =False
                    
                    else:
                        if(i<=n):
                            if(i==n):
                                flagIMP = 0

                            if(Player3.finalstatus[2] !=1):

                                if(Player3.currentposition[2] >= 41):
                                    Player3.makeAMove(Player3.currentposition[2])
                                    Player3.currentposition[2] +=1
                                    i+=1
                                    if track3[Player3.currentposition[2]] == (350,350):
                                        i = n+1

                                elif(Player3.currentposition[2] < 41):
                                    Player3.makeAMove(Player3.currentposition[2])
                                    Player3.currentposition[2] +=1
                                    i+=1
                                else:
                                    p_id = 4
                                    i=1
                                    mouseclicked = False

                            else:
                                renderText("This chess has been to the destination!",850,70,15,BLACK)
                                renderText("Please change another button!.",850,85,15,BLACK)

                        elif(i > n):
                            
                            flag = False
                            for bombPosition in bomb_place:

                                if track3[Player3.currentposition[2]] == bombPosition:
                                    flag = True
                                    bomb_place.remove(bombPosition)  
                                    game_result = Game()                  
                                    
                                    if game_result == -1:
                                        Player3.currentposition[2] = 0   
                                        Player3.backToStart()           
                            
                            for starPosition in star_place:
             
                                if track3[Player3.currentposition[2]] == starPosition:
                                    flag = True
                                    star_place.remove(starPosition)  

                                    Player3.currentposition[2] = 41   
                                    Player3.backToEnd()  
                                    
                            if (n==6) and (flag == False) and (track3[Player3.currentposition[2]] != (350,350)) and (len(bomb_place)<5):
                                bomb_place.append(track3[Player3.currentposition[2]])
                            
                            if (n==1) and (flag == False) and (track3[Player3.currentposition[2]] != (350,350)) and (len(star_place)<5):
                                star_place.append(track3[Player3.currentposition[2]])
                                
                            p_id = 4
                            i=1
                            mouseclicked = False
                            
                if(movebutton ==4):
                    if(flagIMP == 0):
                        diff = (47 - Player3.currentposition[3])
                        flagIMP = 1

                    if(Player3.initialstatus[3] == 0):
                        if(n == 6 or n == 1):

                            Player3.initialstatus[3] = 1
                            initiated3[3] = position3[3]
                            Player3.InitiateStart(4)
                            mouseclicked = False
                            Player3.currentposition[3] = 0

                        else:
                            p_id = 4
                            mouseclicked =False
                    else:
                        if(i<=n):
                            if(i==n):
                                flagIMP = 0

                            if(Player3.finalstatus[3] !=1):
                                if(Player3.currentposition[3] >= 41):
                                    Player3.makeAMove(Player3.currentposition[3])
                                    Player3.currentposition[3] +=1
                                    i+=1
                                    if track3[Player3.currentposition[3]] == (350,350):
                                        i = n+1
                                        
                                elif(Player3.currentposition[3] < 41):
                                    Player3.makeAMove(Player3.currentposition[3])
                                    Player3.currentposition[3] +=1
                                    i+=1
                                else:
                                    p_id = 4
                                    i=1
                                    mouseclicked = False

                            else:
                                renderText("This chess has been to the destination!",850,70,15,BLACK)
                                renderText("Please change another button!.",850,85,15,BLACK)

                        elif(i > n):
                            
                            flag = False
                            for bombPosition in bomb_place:

                                if track3[Player3.currentposition[3]] == bombPosition:
                                    flag = True
                                    bomb_place.remove(bombPosition)  
                                    game_result = Game()                  
                                    
                                    if game_result == -1:
                                        Player3.currentposition[3] = 0   
                                        Player3.backToStart()           
                            
                            for starPosition in star_place:
             
                                if track3[Player3.currentposition[3]] == starPosition:
                                    flag = True
                                    star_place.remove(starPosition)  

                                    Player3.currentposition[3] = 41   
                                    Player3.backToEnd()  
                                    
                            if (n==6) and (flag == False) and (track3[Player3.currentposition[3]] != (350,350)) and (len(bomb_place)<5):
                                bomb_place.append(track3[Player3.currentposition[3]])
                            
                            if (n==1) and (flag == False) and (track3[Player3.currentposition[3]] != (350,350)) and (len(star_place)<5):
                                star_place.append(track3[Player3.currentposition[3]])
                                
                            p_id = 4
                            i=1
                            mouseclicked = False

        elif(p_id == 4):  #当前玩家4正在行动，逻辑同上
            if(Player2.finalstatus[0] == 1 and Player2.finalstatus[1] == 1 and Player2.finalstatus[2] == 1 and Player2.finalstatus[3] == 1):
                p_id = 1
            else:
                if(movebutton ==1):
                    if(flagIMP == 0):
                        diff = (47 - Player4.currentposition[0])
                        flagIMP = 1

                    if(Player4.initialstatus[0] == 0):
                        if(n == 6 or n == 1):

                            Player4.initialstatus[0] = 1
                            initiated4[0] = position4[0]
                            Player4.InitiateStart(1)
                            mouseclicked = False
                            Player4.currentposition[0] = 0
                            
                        else:
                            p_id = 1
                            mouseclicked =False
                    else:
                        if(i<=n):
                            if(i==n):
                                flagIMP = 0

                            if(Player4.finalstatus[0] !=1):
                                if(Player4.currentposition[0] >= 41):
                                    Player4.makeAMove(Player4.currentposition[0])
                                    Player4.currentposition[0] +=1
                                    i+=1
                                    if track4[Player4.currentposition[0]] == (350,350):
                                        i = n+1
                                        
                                elif(Player4.currentposition[0] < 41):
                                    Player4.makeAMove(Player4.currentposition[0])
                                    Player4.currentposition[0] +=1
                                    i+=1
                                else:
                                    p_id = 1
                                    i=1
                                    mouseclicked = False

                            else:
                                renderText("This chess has been to the destination!",850,70,15,BLACK)
                                renderText("Please change another button!.",850,85,15,BLACK)

                        elif(i > n):
                            
                            flag = False
                            for bombPosition in bomb_place:

                                if track4[Player4.currentposition[0]] == bombPosition:
                                    flag = True
                                    bomb_place.remove(bombPosition)  
                                    game_result = Game()                  
                                    
                                    if game_result == -1:
                                        Player4.currentposition[0] = 0   
                                        Player4.backToStart()           
                            
                            for starPosition in star_place:
             
                                if track4[Player4.currentposition[0]] == starPosition:
                                    flag = True
                                    star_place.remove(starPosition)  

                                    Player4.currentposition[0] = 41   
                                    Player4.backToEnd()  
                                    
                            if (n==6) and (flag == False) and (track4[Player4.currentposition[0]] != (350,350)) and (len(bomb_place)<5):
                                bomb_place.append(track4[Player4.currentposition[0]])
                            
                            if (n==1) and (flag == False) and (track4[Player4.currentposition[0]] != (350,350)) and (len(star_place)<5):
                                star_place.append(track4[Player4.currentposition[0]])
                                
                            p_id = 1
                            i=1
                            mouseclicked = False

                if(movebutton ==2):

                    if(flagIMP == 0):
                        diff = (47 - Player4.currentposition[1])
                        flagIMP = 1

                    if(Player4.initialstatus[1] == 0):
                        if(n == 6 or n == 1):

                            Player4.initialstatus[1] = 1
                            initiated4[1] = position4[1]
                            Player4.InitiateStart(2)
                            mouseclicked = False
                            Player4.currentposition[1] = 0

                        else:
                            p_id = 1
                            mouseclicked =False
                    else:
                        if(i<=n):
                            if(i==n):
                                flagIMP = 0

                            if(Player4.finalstatus[1] !=1):
                                if(Player4.currentposition[1] >= 41):
                                    Player4.makeAMove(Player4.currentposition[1])
                                    Player4.currentposition[1] +=1
                                    i+=1
                                    if track4[Player4.currentposition[1]] == (350,350):
                                        i = n+1
                                        
                                elif(Player4.currentposition[1] < 41):
                                    Player4.makeAMove(Player4.currentposition[1])
                                    Player4.currentposition[1] +=1
                                    i+=1
                                else:
                                    p_id = 1
                                    i=1
                                    mouseclicked = False

                            else:
                                renderText("This chess has been to the destination!",850,70,15,BLACK)
                                renderText("Please change another button!.",850,85,15,BLACK)

                        elif(i > n):
                            
                            flag = False
                            for bombPosition in bomb_place:

                                if track4[Player4.currentposition[1]] == bombPosition:
                                    flag = True
                                    bomb_place.remove(bombPosition)  
                                    game_result = Game()                  
                                    
                                    if game_result == -1:
                                        Player4.currentposition[1] = 0   
                                        Player4.backToStart()           
                            
                            for starPosition in star_place:
             
                                if track4[Player4.currentposition[1]] == starPosition:
                                    flag = True
                                    star_place.remove(starPosition)  

                                    Player4.currentposition[1] = 41   
                                    Player4.backToEnd()  
                                    
                            if (n==6) and (flag == False) and (track4[Player4.currentposition[1]] != (350,350)) and (len(bomb_place)<5):
                                bomb_place.append(track4[Player4.currentposition[1]])
                            
                            if (n==1) and (flag == False) and (track4[Player4.currentposition[1]] != (350,350)) and (len(star_place)<5):
                                star_place.append(track4[Player4.currentposition[1]])
                                
                            p_id = 1
                            i=1
                            mouseclicked = False
                            
                if(movebutton ==3):
                    if(flagIMP == 0):
                        diff = (47 - Player4.currentposition[2])
                        flagIMP = 1

                    if(Player4.initialstatus[2] == 0):
                        if(n == 6 or n == 1):

                            Player4.initialstatus[2] = 1
                            initiated4[2] = position4[2]
                            Player4.InitiateStart(3)
                            mouseclicked = False
                            Player4.currentposition[2] = 0

                        else:
                            p_id = 1
                            mouseclicked =False
                   
                    else:
                        if(i<=n):
                            if(i==n):
                                flagIMP = 0

                            if(Player4.finalstatus[2] !=1):

                                if(Player4.currentposition[2] >= 41):
                                    Player4.makeAMove(Player4.currentposition[2])
                                    Player4.currentposition[2] +=1
                                    i+=1
                                    if track4[Player4.currentposition[2]] == (350,350):
                                        i = n+1

                                elif(Player4.currentposition[2] < 41):
                                    Player4.makeAMove(Player4.currentposition[2])
                                    Player4.currentposition[2] +=1
                                    i+=1
                                else:
                                    p_id = 1
                                    i=1
                                    mouseclicked = False

                            else:
                                renderText("This chess has been to the destination!",850,70,15,BLACK)
                                renderText("Please change another button!.",850,85,15,BLACK)

                        elif(i > n):
                            
                            flag = False
                            for bombPosition in bomb_place:

                                if track4[Player4.currentposition[2]] == bombPosition:
                                    flag = True
                                    bomb_place.remove(bombPosition)  
                                    game_result = Game()                  
                                    
                                    if game_result == -1:
                                        Player4.currentposition[2] = 0   
                                        Player4.backToStart()           
                            
                            for starPosition in star_place:
             
                                if track4[Player4.currentposition[2]] == starPosition:
                                    flag = True
                                    star_place.remove(starPosition)  

                                    Player4.currentposition[2] = 41   
                                    Player4.backToEnd()  
                                    
                            if (n==6) and (flag == False) and (track4[Player4.currentposition[2]] != (350,350)) and (len(bomb_place)<5):
                                bomb_place.append(track4[Player4.currentposition[2]])
                            
                            if (n==1) and (flag == False) and (track4[Player4.currentposition[2]] != (350,350)) and (len(star_place)<5):
                                star_place.append(track4[Player4.currentposition[2]])
                                
                            p_id = 1
                            i=1
                            mouseclicked = False
                            
                if(movebutton ==4):
                    if(flagIMP == 0):
                        diff = (47 - Player4.currentposition[3])
                        flagIMP = 1

                    if(Player4.initialstatus[3] == 0):
                        
                        if(n == 6 or n == 1):
                            Player4.initialstatus[3] = 1
                            initiated4[3] = position4[3]
                            Player4.InitiateStart(4)
                            mouseclicked = False
                            Player4.currentposition[3] = 0

                        else:
                            p_id = 1
                            mouseclicked =False
                    
                    else:
                        if(i<=n):
                            if(i==n):
                                flagIMP = 0

                            if(Player4.finalstatus[3] !=1):
                                if(Player4.currentposition[3] >= 41):
                                    Player4.makeAMove(Player4.currentposition[3])
                                    Player4.currentposition[3] +=1
                                    i+=1
                                    if track4[Player4.currentposition[3]] == (350,350):
                                        i = n+1
                                        
                                elif(Player4.currentposition[3] < 41):
                                    Player4.makeAMove(Player4.currentposition[3])
                                    Player4.currentposition[3] +=1
                                    i+=1
                                else:
                                    p_id = 1
                                    i=1
                                    mouseclicked = False

                            else:
                                renderText("This chess has been to the destination!",850,70,15,BLACK)
                                renderText("Please change another button!.",850,85,15,BLACK)

                        elif(i > n):
                            
                            flag = False
                            for bombPosition in bomb_place:
                                if track4[Player4.currentposition[3]] == bombPosition:
                                    flag = True

                                    bomb_place.remove(bombPosition)  
                                    game_result = Game()                  
                                    
                                    if game_result == -1:
                                        Player4.currentposition[3] = 0   
                                        Player4.backToStart()           
                            
                                
                            for starPosition in star_place:
                 
                                if track4[Player4.currentposition[3]] == starPosition:
                                    flag = True
                                    star_place.remove(starPosition)  

                                    Player4.currentposition[3] = 41   
                                    Player4.backToEnd()  
                                    
                            if (n==6) and (flag == False) and (track4[Player4.currentposition[3]] != (350,350)) and (len(bomb_place)<5):
                                bomb_place.append(track4[Player4.currentposition[3]])
                            
                            if (n==1) and (flag == False) and (track4[Player4.currentposition[3]] != (350,350)) and (len(star_place)<5):
                                star_place.append(track4[Player4.currentposition[3]])
                                
                            p_id = 1
                            i=1
                            mouseclicked = False

    Player1.isInHome()
    Player2.isInHome()
    Player3.isInHome()
    Player4.isInHome()

    pygame.display.update()
    fpsClock.tick(FPS)
    

