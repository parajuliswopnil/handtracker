import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)
mp_hand = mp.solutions.hands
hands = mp_hand.Hands()
mpDraw = mp.solutions.drawing_utils
previous_time = 0
current_time = 0

while True:
    success, img = cap.read()
    image_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, landmarks in enumerate(hand_landmarks.landmark):
                # print(id, landmarks)
                height, width, channel = img.shape
                cx, cy = int(landmarks.x * width), int(landmarks.y * height)
                print(id, cx, cy)
                if id == 20:
                    cv.circle(img, (cx, cy), 25, (255, 0, 255), cv.FILLED)
            mpDraw.draw_landmarks(img, hand_landmarks, mp_hand.HAND_CONNECTIONS)

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

    cv.imshow('HandImage', img)
    cv.waitKey(1)
