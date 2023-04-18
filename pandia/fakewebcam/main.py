import pandia.fakewebcam as webcam
import numpy as np
import time
import timeit
import os

from PIL import Image

cam = webcam.FakeWebcam('/dev/video1', 1280, 720)

cam.print_capabilities()

im0 = np.array(Image.open(os.path.expanduser("~/Downloads/doge1.jpg")))
im1 = np.zeros((720, 1280, 3), dtype=np.uint8)

while True:
    t1 = timeit.default_timer()
    cam.schedule_frame(im0)
    t2 = timeit.default_timer()
    print('[1] write time:{}'.format(t2-t1))

    time.sleep(1/60)

    t1 = timeit.default_timer()
    cam.schedule_frame(im1)
    t2 = timeit.default_timer()
    print('[2] write time:{}'.format(t2-t1))
    time.sleep(1/60)
