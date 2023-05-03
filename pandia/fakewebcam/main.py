import pandia.fakewebcam as webcam
import numpy as np
import time
import timeit
import os


def main():
    resolution = 'fhd'
    shape = (3840, 2160)
    if resolution == 'fhd':
        shape = (shape[0] // 2, shape[1] // 2)
    path = os.path.expanduser(f"~/Downloads/drive_{resolution}.yuv")
    cam = webcam.FakeWebcam('/dev/video1', shape[0], shape[1])
    data = np.fromfile(open(path, 'br'), dtype=np.uint8)
    data = data.reshape((-1, shape[0] * 3 // 2, shape[1]))
    print('YUV shape: ', data.shape)
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
        time.sleep(max(0, ts - time.time() + count / fps))


if __name__ == "__main__":
    main()
