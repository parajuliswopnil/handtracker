import cv2 as cv
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, track_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence
        self.mp_hand = mp.solutions.hands
        self.hands = self.mp_hand.Hands(self.mode, self.max_hands, self.detection_confidence, self.track_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB )
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_landmarks, self.mp_hand.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_number=0, draw=True):
        lm_list = list()
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_number]
            for id, landmarks in enumerate(hand.landmark):
                height, width, channel = img.shape
                cx, cy = int(landmarks.x * width), int(landmarks.y * height)
                lm_list.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 5, (255, 0, 255), cv.FILLED)

        return lm_list


def main():
    previous_time = 0
    current_time = 0
    cap = cv.VideoCapture(0)
    hand_detector = HandDetector()  # never make an instance of a class in a loop
    while True:
        success, img = cap.read()
        img = hand_detector.find_hands(img)
        landmark_list = hand_detector.find_position(img)
        if len(landmark_list) != 0:
            print(landmark_list)
        del landmark_list
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

        cv.imshow('HandImage', img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
