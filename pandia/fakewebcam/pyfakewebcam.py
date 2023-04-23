import os
import sys
import fcntl
import timeit
import sys
import time

import numpy as np
import pandia.fakewebcam.v4l2 as _v4l2

cv2_imported = False
try:
    import cv2
    cv2_imported = True
except:
    sys.stderr.write('Warning! opencv could not be imported; performace will be degraded!\n')

class FakeWebcam:
    def __init__(self, video_device, width, height, channels=3, input_pixfmt='RGB'):
        if channels != 3:
            raise NotImplementedError('Code only supports inputs with 3 channels right now. You tried to intialize with {} channels'.format(channels))

        if input_pixfmt != 'RGB':
            raise NotImplementedError('Code only supports RGB pixfmt. You tried to intialize with {}'.format(input_pixfmt))

        if not os.path.exists(video_device):
            sys.stderr.write('\n--- Make sure the v4l2loopback kernel module is loaded ---\n')
            sys.stderr.write('sudo modprobe v4l2loopback devices=1\n\n')
            raise FileNotFoundError('device does not exist: {}'.format(video_device))

        self._channels = channels
        self._video_device = os.open(video_device, os.O_WRONLY | os.O_SYNC)

        self._settings = _v4l2.v4l2_format()
        self._settings.type = _v4l2.V4L2_BUF_TYPE_VIDEO_OUTPUT
        self._settings.fmt.pix.pixelformat = _v4l2.V4L2_PIX_FMT_YUV420
        self._settings.fmt.pix.width = width
        self._settings.fmt.pix.height = height
        self._settings.fmt.pix.field = _v4l2.V4L2_FIELD_NONE
        self._settings.fmt.pix.bytesperline = width * 2
        self._settings.fmt.pix.sizeimage = width * height * 2
        self._settings.fmt.pix.colorspace = _v4l2.V4L2_COLORSPACE_SRGB

        self._buffer = np.zeros((self._settings.fmt.pix.height, 2*self._settings.fmt.pix.width), dtype=np.uint8)

        self._yuv = np.zeros((self._settings.fmt.pix.height, self._settings.fmt.pix.width, 3), dtype=np.uint8)
        self._ones = np.ones((self._settings.fmt.pix.height, self._settings.fmt.pix.width, 1), dtype=np.uint8)
        self._rgb2yuv = np.array([[0.299, 0.587, 0.114, 0],
                                  [-0.168736, -0.331264, 0.5, 128],
                                  [0.5, -0.418688, -0.081312, 128]])

        fcntl.ioctl(self._video_device, _v4l2.VIDIOC_S_FMT, self._settings)


    def print_capabilities(self):
        capability = _v4l2.v4l2_capability()
        print(("get capabilities result", (fcntl.ioctl(self._video_device, _v4l2.VIDIOC_QUERYCAP, capability))))
        print(("capabilities", hex(capability.capabilities)))
        print(("v4l2 driver: {}".format(capability.driver)))


    def schedule_frame(self, frame, sequence=0):
        print(f'Write frame {sequence}')
        os.write(self._video_device, frame)

