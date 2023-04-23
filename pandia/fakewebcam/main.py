import pandia.fakewebcam as webcam
import numpy as np
import time
import timeit
import os


def main():
    cam = webcam.FakeWebcam('/dev/video1', 3840, 2160)
    path = os.path.expanduser("~/Downloads/drive.yuv")
    data = np.fromfile(open(path, 'br'), dtype=np.uint8)
    data = data.reshape((-1, 3840 * 3 // 2, 2160))
    print(data.shape)
    count = 0
    ts = time.time()
    fps = 60
    while True:
        limit = data.shape[0]
        i = count % limit
        if (count // limit) % 2 == 1:
            i = limit - i - 1
        t1 = timeit.default_timer()
        cam.schedule_frame(data[i], count)
        t2 = timeit.default_timer()
        print('write time:{}'.format(t2-t1))
        count += 1
        time.sleep(ts - time.time() + count / fps)


if __name__ == "__main__":
    main()
