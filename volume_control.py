import cv2 as cv
import time
import numpy as np
import handtrackingmodule
import math
from subps import set_master_volume


cam_width, cam_height = 640, 480

cap = cv.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)
previous_time = 0
hand_dect = handtrackingmodule.HandDetector(detection_confidence=0.8)

while True:
    success, img = cap.read()
    img = hand_dect.find_hands(img)
    landmark_list = hand_dect.find_position(img, draw=False)
    if len(landmark_list) != 0:
        print(landmark_list[4], landmark_list[8])
        x1, y1 = landmark_list[4][1], landmark_list[4][2]
        x2, y2 = landmark_list[8][1], landmark_list[8][2]
        cv.circle(img, (landmark_list[4][1], landmark_list[4][2]), 10, (0, 255, 255), cv.FILLED)
        cv.circle(img, (landmark_list[8][1], landmark_list[8][2]), 10, (0, 255, 255), cv.FILLED)
        cv.line(img, (landmark_list[4][1], landmark_list[4][2]),
                (landmark_list[8][1], landmark_list[8][2]), (255, 255, 0), 2)
        cx, cy = int((landmark_list[8][1] + landmark_list[4][1]) / 2), int((landmark_list[8][2] + landmark_list[4][2]) / 2)
        cv.circle(img, (cx, cy), 5, (255, 0, 255), cv.FILLED)

        length_of_the_line = math.hypot(x2 - x1, y2 - y1)
        print(length_of_the_line)
        increasing_percentage = int((length_of_the_line / 160) * 100)
        if increasing_percentage < 15:
            cv.circle(img, (cx, cy), 10, (0, 255, 0), cv.FILLED)
            set_master_volume(0)
        else:
            set_master_volume(increasing_percentage)
        # if length_of_the_line < 40:
        #     cv.circle(img, (cx, cy), 10, (0, 255, 0), cv.FILLED)
        #     cv.putText(img, 'muted', (5, 90), cv.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
        #     set_master_volume(0)
        #
        # if length_of_the_line > 200:
        #     cv.circle(img, (cx, cy), 10, (0, 255, 0), cv.FILLED)
        #     cv.putText(img, 'unmuted', (5, 90), cv.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
        #     set_master_volume(100)

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv.putText(img, f'FPS: {(int(fps))}', (5, 40), cv.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
    cv.imshow('Image', img)
    cv.waitKey(1)
