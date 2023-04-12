import os
import re
import matplotlib.pyplot as plt


class PacketContext(object):
    def __init__(self, rtp_id, payload_type, payload_size, size, sent_at) -> None:
        self.sent_at = sent_at
        self.received_at = -1
        self.rtp_id = rtp_id
        self.payload_type = payload_type
        self.payload_size = payload_size
        self.size = size


class FrameContext(object):
    def __init__(self, frame_id, captured_at) -> None:
        self.frame_id = frame_id
        self.captured_at = captured_at
        self.encoded_at = 0
        self.received_at = 0
        self.decoded_at = 0
        self.codec = 0
        self.encoded_size = 0
        self.encoded_shape = None
        self.rtp_packets: dict = {}
        self.rtp_id_range = None

    def __str__(self) -> str:
        rtp_size = sum([p.payload_size for p in self.rtp_packets.values()])
        return f'Frame id: {self.frame_id}' \
            f', encoded/transmitted size: {self.encoded_size}/{rtp_size}'\
            f', RTP range: {self.rtp_id_range}' \
            f', encode: {self.encoded_at - self.captured_at if self.encoded_at else -1} ms' \
            f', recv: {self.received_at - self.captured_at if self.received_at else -1} ms' \
            f', decode: {self.decoded_at - self.captured_at if self.decoded_at else -1} ms'


class StreamingContext(object):
    def __init__(self) -> None:
        self.frames: dict[int, FrameContext] = {}
        self.frame_ts_map: dict[int, int] = {}
        self.last_frame_id = -1

    def feed(self, data):
        if data['type'] == 'frame_available':
            frame = FrameContext(data['id'], data['capture_ts'])
            self.frames[frame.frame_id] = frame
            self.frame_ts_map[frame.captured_at] = frame.frame_id
            self.last_frame_id = max(self.last_frame_id, frame.frame_id)
        elif data['type'] == 'frame_encoded':
            frame = self.frames[data['id']]
            frame.encoded_at = data['ts']
            frame.codec = data['codec']
            frame.encoded_shape = (data['width'], data['height'])
            frame.encoded_size = data['encoded_size']
        elif data['type'] == 'packet_sent':
            packet = PacketContext(
                data['rtp_id'], data['payload_type'], data['payload_size'], data['size'], data['ts'])
            if packet.payload_type == 125:
                frame_id = self.frame_ts_map[data['frame_ts']]
                frame = self.frames[frame_id]
                frame.rtp_packets[packet.rtp_id] = packet
        elif data['type'] == 'rtp_id_range':
            frame: FrameContext = self.frames[data['frame_id']]
            frame.rtp_id_range = (data['first_rtp_id'], data['last_rtp_id'])
        elif data['type'] == 'frame_decoded':
            for i in sorted(self.frames.keys(), reverse=True):
                frame = self.frames[i]
                if frame.rtp_id_range and frame.rtp_id_range[0] == data['frame_id']:
                    if not frame.decoded_at:
                        frame.decoded_at = data['ts']
                        break
        elif data['type'] == 'frame_received':
            for i in sorted(self.frames.keys(), reverse=True):
                frame = self.frames[i]
                if frame.rtp_id_range and frame.rtp_id_range[0] == data['frame_id']:
                    if not frame.received_at:
                        frame.received_at = data['ts']
                        break


def parse_line(line) -> dict:
    data = {}
    if line.startswith('(video_stream_encoder.cc') and 'RunPostEncode' in line:
        m = re.match(re.compile(
            '.*RunPostEncode, (\\d+), id: (\\d+), codec: (\\d+), size: (\\d+), width: (\\d+), height: (\\d+), .*'), line)
        data['type'] = 'frame_encoded'
        data['ts'] = int(m[1])
        data['id'] = int(m[2])
        data['codec'] = int(m[3])
        data['encoded_size'] = int(m[4])
        data['width'] = int(m[5])
        data['height'] = int(m[6])
    elif line.startswith('(paced_sender.cc') and 'SendRtpPacket' in line:
        m = re.match(re.compile(
            '.*SendRtpPacket, (\\d+), id: (-?\\d+), payload type: (\\d+), payload size: (\\d+), capture ts: (\\d+), size: (\\d+).*'), line)
        data['type'] = 'packet_sent'
        data['ts'] = int(m[1])
        data['rtp_id'] = int(m[2])
        data['payload_type'] = int(m[3])
        data['payload_size'] = int(m[4])
        data['frame_ts'] = int(m[5])
        data['size'] = int(m[6])
    elif line.startswith('(rtp_transport_controller_send.cc') and 'OnTransportFeedback' in line:
        m = re.match(re.compile(
            '.*OnTransportFeedback, (\\d+), id: (-?\\d+), received: (\\d+).*'), line)
        data['type'] = 'packet_acked'
        data['ts'] = int(m[1])
        data['rtp_id'] = int(m[2])
        data['received'] = int(m[3])
    elif line.startswith('(video_stream_encoder.cc:') and 'OnFrame' in line:
        m = re.match(re.compile(
            '.*OnFrame, (\\d+), id: (-?\\d+), captured at: (\\d+).*'), line)
        data['type'] = 'frame_available'
        data['ts'] = int(m[1])
        data['id'] = int(m[2])
        data['capture_ts'] = int(m[3])
    elif line.startswith('(rtp_sender_video.cc:') and 'SendVideo' in line:
        m = re.match(re.compile(
            '.*SendVideo, frame id: (\\d+), first RTP id: (\\d+), last RTP id: (\\d+).*'), line)
        data['type'] = 'rtp_id_range'
        data['frame_id'] = int(m[1])
        data['first_rtp_id'] = int(m[2])
        data['last_rtp_id'] = int(m[3])
    elif line.startswith('(rtcp_receiver.cc:') and 'Receive RTCP app' in line:
        if 'FrameRecv' in line:
            m = re.match(re.compile(
                '.*Receive RTCP app, (\\d+), name: FrameRecv, frame id: (\\d+).*'), line)
            data['type'] = 'frame_received'
            data['ts'] = int(m[1])
            data['frame_id'] = int(m[2])
        elif 'FrameDecode' in line:
            m = re.match(re.compile(
                '.*Receive RTCP app, (\\d+), name: FrameDecode, frame id: (\\d+).*'), line)
            data['type'] = 'frame_decoded'
            data['ts'] = int(m[1])
            data['frame_id'] = int(m[2])
    return data


def analyze_stream(context: StreamingContext) -> None:
    frame_id_list = list(sorted(context.frames.keys()))
    data_frame_delay = []
    data_packet_delay = []
    for frame_id in frame_id_list:
        frame: FrameContext = context.frames[frame_id]
        data_frame_delay.append((frame_id, frame.encoded_at - frame.captured_at if frame.encoded_at > 0 else 0,
                                 frame.received_at - frame.captured_at if frame.received_at > 0 else 0,
                                 frame.decoded_at - frame.captured_at if frame.decoded_at > 0 else 0))
        print(frame)

    plt.close()
    plt.plot([d[0] for d in data_frame_delay],
             [d[3] for d in data_frame_delay])
    plt.plot([d[0] for d in data_frame_delay],
             [d[2] for d in data_frame_delay])
    plt.plot([d[0] for d in data_frame_delay],
             [d[1] for d in data_frame_delay],)
    plt.legend(['Encoding', 'Transmission', 'Decoding'])
    plt.xlabel('Frame ID')
    plt.ylabel('Delay (ms)')
    plt.ylim([0, 300])
    output_dir = os.path.expanduser('~/Workspace/Pandia/results')
    plt.savefig(os.path.join(output_dir, 'delay.pdf'))


def main() -> None:
    log_dir = os.path.expanduser("~/Workspace/Pandia/AlphaRTC/results")
    sender_log = os.path.join(log_dir, 'sender.log')
    receiver_log = os.path.join(log_dir, 'receiver.log')
    context = StreamingContext()
    for line in open(sender_log).readlines():
        log = parse_line(line)
        if log:
            context.feed(log)
    analyze_stream(context)


if __name__ == "__main__":
    main()
