from collections import deque
from typing import Tuple, Union

from pandia.context.frame_context import FrameContext
from pandia.context.packet_context import PacketContext


class MonitorBlockData(object):
    def __init__(self, ts_fn, val_fn, duration=1, val_checker=lambda v: v >= 0, ts_offset=0) -> None:
        self.duration = duration
        self.data = deque()
        self.num = 0
        self.sum = 0
        self.ts_fn = ts_fn
        self.val_fn = val_fn
        self.val_checker = val_checker
        self.ts_offset = ts_offset

    def ts(self, val: Union[FrameContext, PacketContext]):
        return self.ts_fn(val) - self.ts_offset

    def append(self, val: Union[FrameContext, PacketContext, Tuple], ts):
        if self.val_checker(self.val_fn(val)):
            self.data.append(val)
            self.num += 1
            self.sum += self.val_fn(val) 
        while self.num > 0 and self.ts(self.data[0]) < ts - self.duration:
            val = self.data.popleft()
            self.num -= 1
            self.sum -= self.val_fn(val)

    def avg(self):
        return self.sum / self.num if self.num > 0 else 0
