import pandia.fakewebcam as webcam
import numpy as np
import os
import cv2


def main(width=1080):
    path = os.path.expanduser("~/Downloads/drive.mp4")
    target_path = path.replace('.mp4', f'_{width}p.yuv')
    if os.path.exists(target_path):
        return
    cap = cv2.VideoCapture(path)
    frames = 100
    count = 0
    scale = width / 2160
    data = []
    with open(target_path, 'bw+') as f:
        while cap.isOpened() and count < frames:
            print(f'Process frame #{count}')
            if count <= frames // 2:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (int(frame.shape[0] * scale), int(frame.shape[1] * scale)))
                    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
                    f.write(yuv.tobytes())
                    if count != frames // 2:
                        data.append(yuv.tobytes())
            else:
                f.write(data.pop())
            count += 1


if __name__ == "__main__":
    for w in [144, 240, 360, 720, 960, 1080, 1440, 2160]:
        main(w)
