import os
import re

from matplotlib import pyplot as plt
import numpy as np
from pandia import RESULTS_PATH

from pandia.log_analyzer import OUTPUT_DIR


class Frame:
    def __init__(self, frame_id) -> None:
        self.frame_id = frame_id
        self.received_at = None
        self.decoded_at = None
        self.decoding_at = None


class Packet:
    def __init__(self, rtp_id, seq, ts) -> None:
        self.rtp_id = rtp_id
        self.seq = seq
        self.recv_ts = ts


class Stream:
    def __init__(self) -> None:
        self.frames = {}
        self.packets = {}


def parse_line(line, stream: Stream) -> None:
    if line.startswith('(libvpx_vp8_decoder.cc') and 'Finish decoding' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] Finish decoding, frame first rtp: (\\d+).*'), line)
        ts = int(m[1]) / 1000
        frame_id = int(m[2])
        frame = stream.frames.get(frame_id, None)
        if frame:
            frame.decoded_at = ts
    elif 'Frame decoded' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] Frame decoded, frame id: (\\d+), first rtp sequence: (\\d+).*'), line)
        ts = int(m[1]) / 1000
        frame_id = int(m[3])
        frame = stream.frames.get(frame_id, None)
        if frame:
            frame.decoded_at = ts
    elif 'Frame received' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] Frame received, id: (\\d+), first rtp seq: (\\d+).*'), line)
        ts = int(m[1]) / 1000
        frame_id = int(m[3])
        frame = Frame(frame_id)
        frame.received_at = ts
        stream.frames[frame_id] = frame
    elif 'Rtp packet received' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] Rtp packet received, id: (\\d+), sequence number: (\\d+).*'), line)
        ts = int(m[1]) / 1000
        seq = int(m[2])
        rtp_id = int(m[3])
        packet = Packet(rtp_id, seq, ts)
        stream.packets[rtp_id] = packet
    elif line.startswith('(libvpx_vp8_decoder.cc') and 'Start decoding' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] Start decoding, frame first rtp: (\\d+).*'), line)
        ts = int(m[1]) / 1000
        frame_id = int(m[2])
        frame = stream.frames.get(frame_id, None)
        if frame:
            frame.decoding_at = ts
    elif 'Start decoding' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] Start decoding, frame id: (\\d+), first rtp sequence: (\\d+).*'), line)
        ts = int(m[1]) / 1000
        frame_id = int(m[3])
        frame = stream.frames.get(frame_id, None)
        if frame:
            frame.decoding_at = ts

def analyze(stream: Stream, output_dir=OUTPUT_DIR) -> None:
    print("========== STATISTICS [RECEIVER] ==========")
    if not stream.frames:
        return
    ids = list(sorted(stream.frames.keys()))
    x = []
    y1 = []
    y2 = []
    print(f'Number of frames: {len(ids)}')
    ts_min = stream.frames[ids[0]].received_at
    for frame_id in ids:
        frame = stream.frames[frame_id]
        if frame.decoding_at:
            x.append(frame.received_at - ts_min)
            y1.append(frame.decoding_at - frame.received_at)
            if frame.decoded_at:
                y2.append(frame.decoded_at - frame.received_at)
            else:
                y2.append(0)
    y1 = np.array(y1)
    y2 = np.array(y2)
    plt.close()
    plt.plot(x, y1 * 1000, '.')
    plt.plot(x, y2 * 1000, 'x')
    plt.xlabel('Frame receiving time (s)')
    plt.ylabel('Delay (ms)')
    plt.legend(['Queuing Delay', 'Decoding Delay'])
    plt.savefig(os.path.join(output_dir, 'mea-delay-frame-receiver.pdf'))


def main(result_path=os.path.join(RESULTS_PATH, 'eval_static')) -> None:
    receiver_log = os.path.join(result_path, 'eval_receiver.log')
    stream = Stream()
    for line in open(receiver_log).readlines()[:-1]:  # Ignore the last line because it may be incomplete
        line = line.strip()
        if line:
            try:
                parse_line(line, stream)
            except Exception as e:
                print(f"Error parsing line: {line}")
                raise e
    analyze(stream)




if __name__ == "__main__":
    main()
