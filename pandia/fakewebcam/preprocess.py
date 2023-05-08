import pandia.fakewebcam as webcam
import numpy as np
import os
import cv2


def main(width=1080):
    path = os.path.expanduser("~/Downloads/drive.mp4")
    cap = cv2.VideoCapture(path)
    frames = 360
    count = 0
    scale = width / 2160
    target_path = path.replace('.mp4', f'_{width}p.yuv')
    with open(target_path, 'bw+') as f:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (int(frame.shape[0] * scale), int(frame.shape[1] * scale)))
                yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
            print(f'Process frame #{count}')
            f.write(yuv.tobytes())
            count += 1
            if count >= frames:
                break


if __name__ == "__main__":
    for w in [2160, 1440, 1080, 960, 480, 360]:
        main(w)
