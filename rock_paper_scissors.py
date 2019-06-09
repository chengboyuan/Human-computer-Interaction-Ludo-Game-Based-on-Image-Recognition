import cv2
import numpy as np
import copy
import math
import random
# you need to install open cv "pip install opencv-python"
#hhhhhhhhhhhhhhhhhhh
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

            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
            # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        
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
    # print p1,' ',pc,'\n'
    print("you are ", play[p1], ",and the computer is ", play[pc], "\n")
    #to judge the winner of the game.
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

Game()