import pandia.fakewebcam as webcam
import numpy as np
import os
import cv2


def main():
    path = os.path.expanduser("~/Downloads/drive.mp4")
    target_path = path.replace('.mp4', '.yuv')
    cap = cv2.VideoCapture(path)
    frames = 360
    count = 0
    with open(target_path, 'bw+') as f:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
            print(f'Process frame #{count}')
            f.write(yuv.tobytes())
            count += 1
            if count >= frames:
                break


if __name__ == "__main__":
    main()
