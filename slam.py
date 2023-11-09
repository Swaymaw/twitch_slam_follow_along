import cv2 
from display import Display
import numpy as np
from extractor import FeatureExtractor

W, H = 1920//2, 1080//2

display = Display(W, H)

fe = FeatureExtractor()

def process_frame(img):
    img = cv2.resize(img, (W, H))
    matches = fe.extract(img)

    for pt1, pt2 in matches:
        u1, v1 = map(lambda x: int(round(x)), pt1.pt)
        u2, v2 = map(lambda x: int(round(x)), pt2.pt)

        cv2.circle(img, (u1, v1), color=(0, 255, 0), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color=(255, 0, 0))
    display.paint(img)

if __name__ == '__main__':
    cap = cv2.VideoCapture('vids/test.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break
