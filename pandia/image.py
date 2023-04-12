import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def read_yuv():
    # width, height = 320, 240
    width, height = 3840, 2160
    frame_size = height * 3 // 2 * width
    path = '/home/lix16/Workspace/Pandia/AlphaRTC/bin/testmedia/drive.yuv'
    # path = '/home/lix16/Workspace/Pandia/AlphaRTC/bin/testmedia/test.yuv'
    data = open(path, 'rb').read()
    data = np.frombuffer(data, np.uint8)
    n_frames = len(data) // frame_size
    for i in range(n_frames):
        yuv = data[i * frame_size: (i + 1) * frame_size]
        yuv = yuv.reshape(height * 3 // 2, width)
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        yuv.reshape(height * 3 // 2, width)
        plt.imshow(bgr)
        plt.savefig(os.path.expanduser(f'~/Downloads/frame{i}.jpg'))
        plt.close()


def read_mp4():
    path = os.path.expanduser("~/Downloads/drive.mp4")
    cap = cv2.VideoCapture(path)
    width = 3840
    height = 2160
    frames = 1000
    frame_size = height * 3 // 2 * width
    data = np.empty(shape=(frame_size * frames, ), dtype=np.uint8)
    i = 0
    while cap.isOpened() and i < frames:
        ret, frame = cap.read()
        if not ret:
            continue
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420).flatten()
        data[i * frame_size: (i + 1) * frame_size] = yuv
        i += 1
    with open('/home/lix16/Workspace/Pandia/AlphaRTC/bin/testmedia/drive.yuv', 'wb+') as f:
        np.save(f, data)



def main():
    read_mp4()
    # read_yuv()


if __name__ == '__main__':
    main()
