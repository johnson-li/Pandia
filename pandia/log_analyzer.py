import os
import re
import matplotlib.pyplot as plt
import numpy as np

CODEC_NAMES = ['Generic', 'VP8', 'VP9', 'AV1', 'H264', 'Multiplex']

class PacketContext(object):
    def __init__(self, rtp_id, payload_type, size, sent_at) -> None:
        self.sent_at = sent_at
        self.acked_at = -1
        self.rtp_id = rtp_id
        self.payload_type = payload_type
        self.size = size
        self.received = None


class FrameContext(object):
    def __init__(self, frame_id, captured_at, width, height) -> None:
        self.frame_id = frame_id
        self.captured_at = captured_at
        self.width = width
        self.height = height
        self.encoded_at = 0
        self.assembled_at = 0
        self.decoded_at = 0
        self.codec = None
        self.encoded_size = 0
        self.encoded_shape = None
        self.rtp_packets: dict = {}
        self.rtp_id_range = [1000000, 0]
        self.rtp_packets_num = 0

    def last_rtp_send_ts(self):
        if self.rtp_packets:
            return max([p.sent_at for p in self.rtp_packets.values()if p])
        return None

    def last_rtp_recv_ts(self):
        if self.rtp_packets:
            return max([p.acked_at for p in self.rtp_packets.values() if p])
        return None

    def __str__(self) -> str:
        rtp_size = sum([p.size for p in self.rtp_packets.values() if p])
        return f'[{self.codec}] Frame id: {self.frame_id}' \
            f', encoded/transmitted size: {self.encoded_size}/{rtp_size}'\
            f', RTP range: {self.rtp_id_range} ({self.rtp_packets_num})' \
            f', encode: {self.encoded_at - self.captured_at if self.encoded_at else -1} ms' \
            f', send: {self.last_rtp_send_ts() - self.captured_at if self.last_rtp_send_ts() else -1} ms' \
            f', recv: {self.last_rtp_recv_ts() - self.captured_at if self.last_rtp_recv_ts() else -1} ms' \
            f', assembly: {self.assembled_at - self.captured_at if self.assembled_at else -1} ms' \
            f', decode: {self.decoded_at - self.captured_at if self.decoded_at else -1} ms'


class StreamingContext(object):
    def __init__(self) -> None:
        self.frames: dict[int, FrameContext] = {}
        self.packets = {}
        self.frame_ts_map: dict[int, int] = {}
        self.packet_id_map = {}
        self.last_frame_id = -1


def parse_line(line, context: StreamingContext) -> dict:
    data = {}
    if line.startswith('(video_capture_impl.cc') and 'FrameCaptured' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] FrameCaptured, id: (\\d+), width: (\\d+), height: (\\d+), .*'), line)
        ts = int(m[1]) 
        frame_id = int(m[2])
        width = int(m[3])
        height = int(m[4])
        frame = FrameContext(frame_id, ts, width, height)
        context.frames[frame_id] = frame
        context.frame_ts_map[frame.captured_at] = frame.frame_id
        context.last_frame_id = max(context.last_frame_id, frame_id)
    elif line.startswith('(video_stream_encoder.cc') and 'Frame encoded' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] Frame encoded, id: (\\d+), codec: (\\d+), size: (\\d+), width: (\\d+), height: (\\d+), .*'), line)
        ts = int(m[1])
        frame_id = int(m[2])
        codec = CODEC_NAMES[int(m[3])]
        encoded_size = int(m[4])
        width = int(m[5])
        height = int(m[6])
        frame = context.frames[frame_id]
        frame.encoded_at = ts
        frame.codec = codec
        frame.encoded_shape = (width, height)
        frame.encoded_size = encoded_size
    elif line.startswith('(packet_router.cc') and 'Assign RTP id' in line:
        m = re.match(re.compile(
            '.*Assign RTP id, id: (\\d+), frame id: (\\d+).*'), line)
        rtp_id = int(m[1])
        frame_id = int(m[2])
        if frame_id > 0:
            frame: FrameContext = context.frames[frame_id]
            frame.rtp_packets[rtp_id] = None
            frame.rtp_id_range[0] = min(frame.rtp_id_range[0], rtp_id)
            frame.rtp_id_range[1] = max(frame.rtp_id_range[1], rtp_id)
    elif line.startswith('(rtp_rtcp_impl2.cc') and 'Assign sequence number' in line:
        m = re.match(re.compile(
            '.*Assign sequence number, id: (\\d+), sequence number: (\\d+).*'), line)
        rtp_id = int(m[1])
        sequence_number = int(m[2])
        context.packet_id_map[sequence_number] = rtp_id
    elif line.startswith('(rtp_transport_controller_send.cc') and 'OnSentPacket' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] OnSentPacket, id: (-?\\d+), type: (\\d+), size: (\\d+).*'), line)
        ts = int(m[1])
        rtp_id = int(m[2])
        rtp_type = int(m[3])
        size = int(m[4])
        if rtp_id >= 0:
            packet = PacketContext(rtp_id, rtp_type, size, ts)
            if packet.payload_type == 1:
                context.packets[packet.rtp_id] = packet
                for i in range(context.last_frame_id, 0, -1):
                    frame = context.frames[i]
                    if rtp_id in frame.rtp_packets:
                        frame.rtp_packets[rtp_id] = packet
    elif line.startswith('(transport_feedback_demuxer.cc:') and 'Packet acked' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] Packet acked, id: (\\d+), received: (\\d+), delta: (\\d+).*'), line)
        ts = int(m[1])
        rtp_id = int(m[2])
        received = int(m[3])
        delta = int(m[4])
        packet: PacketContext = context.packets[rtp_id]
        packet.acked_at = ts
        packet.received = received == 1
    elif line.startswith('(rtp_sender_video.cc:') and 'SendVideo' in line:
        m = re.match(re.compile(
            '.*SendVideo, frame id: (\\d+), number of RTP packets: (\\d+).*'), line)
        frame_id = int(m[1])
        rtp_packets = int(m[2])
        frame = context.frames[frame_id]
        frame.rtp_packets_num = rtp_packets
    elif line.startswith('(rtcp_receiver.cc') and 'Frame decoding acked' in line:
        m = re.match(re.compile('.*\\[(\\d+)\\] Frame decoding acked, id: (\\d+).*'), line)
        ts = int(m[1])
        rtp_sequence = int(m[2])
        rtp_id = context.packet_id_map[rtp_sequence]
        for i in range(context.last_frame_id, 0, -1):
            frame: FrameContext = context.frames[i]
            if frame.rtp_id_range[0] == rtp_id:
                frame.decoded_at = ts
                break
            if frame.rtp_id_range[0] < rtp_id:
                break
    elif line.startswith('(rtcp_receiver.cc') and 'Frame reception acked' in line:
        m = re.match(re.compile('.*\\[(\\d+)\\] Frame reception acked, id: (\\d+).*'), line)
        ts = int(m[1])
        rtp_sequence = int(m[2])
        rtp_id = context.packet_id_map[rtp_sequence]
        for i in range(context.last_frame_id, 0, -1):
            frame: FrameContext = context.frames[i]
            if frame.rtp_id_range[0] == rtp_id:
                frame.assembled_at = ts
                break
            if frame.rtp_id_range[0] < rtp_id:
                break
    return data


def analyze_frame(context: StreamingContext) -> None:
    frame_id_list = list(sorted(context.frames.keys()))
    data_frame_delay = []
    data_packet_delay = []
    for frame_id in frame_id_list:
        frame: FrameContext = context.frames[frame_id]
        data_frame_delay.append((frame_id, frame.encoded_at - frame.captured_at if frame.encoded_at > 0 else 0,
                                 frame.assembled_at - frame.captured_at if frame.assembled_at > 0 else 0,
                                 frame.decoded_at - frame.captured_at if frame.decoded_at > 0 else 0))
        print(frame)

    plt.close()
    plt.plot([d[0] for d in data_frame_delay],
             [d[3] for d in data_frame_delay])
    plt.plot([d[0] for d in data_frame_delay],
             [d[2] for d in data_frame_delay])
    plt.plot([d[0] for d in data_frame_delay],
             [d[1] for d in data_frame_delay],)
    plt.legend(['Decoding', 'Transmission', 'Encoding'])
    plt.xlabel('Frame ID')
    plt.ylabel('Delay (ms)')
    plt.ylim([0, 300])
    output_dir = os.path.expanduser('~/Workspace/Pandia/results')
    plt.savefig(os.path.join(output_dir, 'delay-frame.pdf'))


def analyze_packet(context: StreamingContext) -> None:
    data = []
    for pkt in sorted(context.packets.values(), key=lambda x: x.sent_at):
        pkt: PacketContext = pkt
        data.append((pkt.sent_at, pkt.acked_at -
                    pkt.sent_at if pkt.acked_at > 0 else 0))
    plt.close()
    start_ts = min([d[0] for d in data])
    plt.plot([(d[0] - start_ts)/1000 for d in data], [d[1] for d in data], 'x')
    plt.xlabel('Timestamp (ms)')
    plt.ylabel('RTT (s)')
    # plt.xlim([0, 10])
    output_dir = os.path.expanduser('~/Workspace/Pandia/results')
    plt.savefig(os.path.join(output_dir, 'delay-packet.pdf'))
    cdf_x = list(sorted([d[1] for d in data]))
    cdf_y = np.arange(len(cdf_x)) / len(cdf_x)
    plt.close()
    plt.plot(cdf_x, cdf_y)
    plt.xlabel('Packet ACK delay (ms)')
    plt.ylabel('CDF')
    plt.ylim([0, 1])
    plt.xlim([0, max(cdf_x)])
    plt.savefig(os.path.join(output_dir, 'delay-cdf-packet.pdf'))


def analyze_stream(context: StreamingContext) -> None:
    analyze_frame(context)
    analyze_packet(context)


def main() -> None:
    log_dir = os.path.expanduser("~/Workspace/Pandia/results")
    sender_log = os.path.join(log_dir, 'sender.log')
    receiver_log = os.path.join(log_dir, 'receiver.log')
    context = StreamingContext()
    for line in open(sender_log).readlines():
        parse_line(line, context)
    analyze_stream(context)


if __name__ == "__main__":
    main()
