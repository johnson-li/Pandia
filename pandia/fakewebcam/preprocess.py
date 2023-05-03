import pandia.fakewebcam as webcam
import numpy as np
import os
import cv2


def main():
    path = os.path.expanduser("~/Downloads/drive.mp4")
    cap = cv2.VideoCapture(path)
    frames = 360
    count = 0
    downscale = 1
    if downscale == 1:
        target_path = path.replace('.mp4', '_4k.yuv')
    if downscale == 2:
        target_path = path.replace('.mp4', '_fhd.yuv')
    with open(target_path, 'bw+') as f:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (int(frame.shape[0] / downscale), int(frame.shape[1] / downscale)))
                yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
            print(f'Process frame #{count}')
            f.write(yuv.tobytes())
            count += 1
            if count >= frames:
                break


if __name__ == "__main__":
    main()
