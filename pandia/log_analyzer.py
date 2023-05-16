import os
import re
import threading
import matplotlib.pyplot as plt
import numpy as np

from pandia import RESULTS_PATH, DIAGRAMS_PATH

CODEC_NAMES = ['Generic', 'VP8', 'VP9', 'AV1', 'H264', 'Multiplex']
OUTPUT_DIR = DIAGRAMS_PATH


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
        if list(filter(lambda x: x, self.rtp_packets.values())):
            return max([p.sent_at for p in self.rtp_packets.values()if p])
        return None

    def last_rtp_recv_ts(self):
        if list(filter(lambda x: x, self.rtp_packets.values())):
            return max([p.acked_at for p in self.rtp_packets.values() if p])
        return None

    def encoding_delay(self):
        return self.encoded_at - self.captured_at if self.encoded_at and self.captured_at else -1

    def transmission_delay(self):
        return self.assembled_at - self.encoded_at if self.assembled_at and self.encoded_at else -1

    def decoding_delay(self):
        return self.decoded_at - self.assembled_at if self.decoded_at and self.assembled_at else -1

    def received(self):
        return self.assembled_at >= 0

    def __str__(self) -> str:
        rtp_size = sum([p.size for p in self.rtp_packets.values() if p])
        return f'[{self.codec}] Frame id: {self.frame_id} {self.encoded_shape}' \
            f', encoded/transmitted size: {self.encoded_size}/{rtp_size}'\
            f', RTP range: {self.rtp_id_range} ({self.rtp_packets_num})' \
            f', encode: {self.encoded_at - self.captured_at if self.encoded_at else -1} ms' \
            f', send: {self.last_rtp_send_ts() - self.captured_at if self.last_rtp_send_ts() else -1} ms' \
            f', recv: {self.last_rtp_recv_ts() - self.captured_at if self.last_rtp_recv_ts() else -1} ms' \
            f', assembly: {self.assembled_at - self.captured_at if self.assembled_at else -1} ms' \
            f', decode: {self.decoded_at - self.captured_at if self.decoded_at else -1} ms'


class StreamingContext(object):
    def __init__(self) -> None:
        self.start_ts = 0
        self.frames: dict[int, FrameContext] = {}
        self.packets: dict[int, PacketContext] = {}
        self.packet_id_map = {}
        self.networking = NetworkContext()
        self.bitrate_data = []
        self.fps_data = []
        self.last_captured_frame_id = 0
        self.last_decoded_frame_id = 0
        self.last_egress_packet_id = 0
        self.last_acked_packet_id = 0

    def latest_frames(self, duration=1):
        res = []
        for i in range(self.last_captured_frame_id, 0, -1):
            if self.frames[i].captured_at >= self.frames[self.last_captured_frame_id].captured_at - duration:
                res.append(self.frames[i])
            else:
                break
        return res

    def latest_egress_packets(self, duration=1):
        res = []
        for i in range(self.last_egress_packet_id, 0, -1):
            if self.packets[i].sent_at >= self.packets[self.last_egress_packet_id].sent_at - duration:
                res.append(self.packets[i])
            else:
                break
        return res

    def latest_acked_packets(self, duration=1):
        res = []
        for i in range(self.last_acked_packet_id, 0, -1):
            if self.packets[i].acked_at >= self.packets[self.last_acked_packet_id].acked_at - duration:
                res.append(self.packets[i])
            else:
                break
        return res


class NetworkContext(object):
    def __init__(self) -> None:
        self.pacing_rate_data = []
        self.pacing_burst_interval_data = []


def parse_line(line, context: StreamingContext) -> dict:
    data = {}
    if line.startswith('(video_capture_impl.cc') and 'FrameCaptured' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] FrameCaptured, id: (\\d+), width: (\\d+), height: (\\d+), .*'), line)
        ts = int(m[1]) / 1000
        frame_id = int(m[2])
        width = int(m[3])
        height = int(m[4])
        frame = FrameContext(frame_id, ts, width, height)
        context.last_captured_frame_id = frame_id
        context.frames[frame_id] = frame
    elif line.startswith('(main.cc') and 'Program started' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] Program started.*'), line)
        ts = int(m[1]) / 1000
        context.start_ts = ts
    elif line.startswith('(video_stream_encoder.cc') and 'Frame encoded' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] Frame encoded, id: (\\d+), codec: (\\d+), size: (\\d+), width: (\\d+), height: (\\d+), .*'), line)
        ts = int(m[1]) / 1000
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
        ts = int(m[1]) / 1000
        rtp_id = int(m[2])
        rtp_type = int(m[3])
        size = int(m[4])
        if rtp_id >= 0:
            packet = PacketContext(rtp_id, rtp_type, size, ts)
            if packet.payload_type == 1:
                context.packets[packet.rtp_id] = packet
                for i in range(context.last_captured_frame_id, 0, -1):
                    frame = context.frames[i]
                    if rtp_id in frame.rtp_packets:
                        frame.rtp_packets[rtp_id] = packet
            context.last_egress_packet_id = max(
                rtp_id, context.last_egress_packet_id)
    elif line.startswith('(transport_feedback_demuxer.cc:') and 'Packet acked' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] Packet acked, id: (\\d+), received: (\\d+), delta: (-?\\d+).*'), line)
        ts = int(m[1]) / 1000
        rtp_id = int(m[2])
        received = int(m[3])
        delta = int(m[4])
        packet: PacketContext = context.packets[rtp_id]
        packet.acked_at = ts
        packet.received = received == 1
        context.last_acked_packet_id = max(
            rtp_id, context.last_acked_packet_id)
    elif line.startswith('(rtp_sender_video.cc:') and 'SendVideo' in line:
        m = re.match(re.compile(
            '.*SendVideo, frame id: (\\d+), number of RTP packets: (\\d+).*'), line)
        frame_id = int(m[1])
        rtp_packets = int(m[2])
        frame = context.frames[frame_id]
        frame.rtp_packets_num = rtp_packets
    elif line.startswith('(rtcp_receiver.cc') and 'Frame decoding acked' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] Frame decoding acked, id: (\\d+).*'), line)
        ts = int(m[1]) / 1000
        rtp_sequence = int(m[2])
        rtp_id = context.packet_id_map.get(rtp_sequence, None)
        if rtp_id:
            for i in range(context.last_captured_frame_id, 0, -1):
                frame: FrameContext = context.frames[i]
                if frame.rtp_id_range[0] == rtp_id:
                    frame.decoded_at = ts
                    context.last_decoded_frame_id = max(
                        frame.frame_id, context.last_decoded_frame_id)
                    break
                if frame.rtp_id_range[0] < rtp_id:
                    break
    elif line.startswith('(rtcp_receiver.cc') and 'Frame reception acked' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] Frame reception acked, id: (\\d+).*'), line)
        ts = int(m[1]) / 1000
        rtp_sequence = int(m[2])
        rtp_id = context.packet_id_map[rtp_sequence]
        for i in range(context.last_captured_frame_id, 0, -1):
            frame: FrameContext = context.frames[i]
            if frame.rtp_id_range[0] == rtp_id:
                frame.assembled_at = ts
                break
            if frame.rtp_id_range[0] < rtp_id:
                break
    elif line.startswith('(task_queue_paced_sender.cc') and 'SetPacingRates' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] SetPacingRates, pacing rate: (\\d+) kbps, pading rate: (\\d+) kbps.*'), line)
        ts = int(m[1]) / 1000
        pacing_rate = int(m[2])
        padding_rate = int(m[3])
        context.networking.pacing_rate_data.append(
            [ts, pacing_rate, padding_rate])
    elif line.startswith('(video_stream_encoder.cc') and 'SetRates, ' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] SetRates, bitrate: (\\d+) kbps, framerate: (\\d+).*'), line)
        ts = int(m[1]) / 1000
        bitrate = int(m[2])
        fps = int(m[3])
        context.bitrate_data.append([ts, bitrate])
        context.fps_data.append([ts, fps])
    return data


def analyze_frame(context: StreamingContext) -> None:
    frame_id_list = list(sorted(context.frames.keys()))
    data_frame_delay = []
    lost_frames = []
    started = False
    last_frame_id = -1
    for frame_id in frame_id_list:
        frame: FrameContext = context.frames[frame_id]
        if frame.codec:
            started = True
            last_frame_id = frame_id
            data_frame_delay.append({'id': frame_id,
                                     'ts': frame.captured_at,
                                     'encoded by': frame.encoded_at - frame.captured_at if frame.encoded_at > 0 else 0,
                                     'paced by': frame.last_rtp_send_ts() - frame.captured_at if frame.last_rtp_send_ts() > 0 else 0,
                                     'assembled by': frame.assembled_at - frame.captured_at if frame.assembled_at > 0 else 0,
                                     'decoded by': frame.decoded_at - frame.captured_at if frame.decoded_at > 0 else 0})
        elif started:
            lost_frames.append(frame.captured_at)
    lost_frames = [f for f in lost_frames if f <= last_frame_id]
    plt.close()
    ylim = 0
    for i in ['decoded by', 'assembled by', 'paced by', 'encoded by']:
        x = np.array([d['ts'] - context.start_ts for d in data_frame_delay])
        y = np.array([d[i] for d in data_frame_delay]) * 1000
        print(f'Median: {i} {np.median(y)} ms')
        if not ylim:
            ylim = np.percentile(y, 50)
        indexes = (y > 0).nonzero()
        plt.plot(x[indexes], y[indexes])
    plt.plot([i - context.start_ts for i in lost_frames], [10 for _ in lost_frames], 'x')
    plt.legend(['Decoding', 'Transmission', 'Pacing', 'Encoding', 'Lost'])
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Delay (ms)')
    plt.ylim([0, ylim * 1.8])
    plt.savefig(os.path.join(OUTPUT_DIR, 'delay-frame.pdf'))
    plt.close()
    x = []
    y = []
    for frame_id in frame_id_list:
        frame: FrameContext = context.frames[frame_id]
        if (frame.encoded_size):
            x.append(frame_id)
            y.append(frame.encoded_size / 1024)
    plt.plot(x, y)
    plt.plot(lost_frames, [10 for _ in lost_frames], 'x')
    plt.xlabel('Frame ID')
    plt.ylabel('Encoded size (KB)')
    plt.savefig(os.path.join(OUTPUT_DIR, 'size-frame.pdf'))


def analyze_packet(context: StreamingContext) -> None:
    data = []
    for pkt in sorted(context.packets.values(), key=lambda x: x.sent_at):
        pkt: PacketContext = pkt
        data.append((pkt.sent_at, pkt.acked_at -
                    pkt.sent_at if pkt.acked_at > 0 else 0))
    plt.close()
    plt.plot([(d[0] - context.start_ts)
             for d in data], [d[1] * 1000 for d in data], 'x')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('RTT (ms)')
    # plt.xlim([0, 10])
    plt.savefig(os.path.join(OUTPUT_DIR, 'delay-packet.pdf'))
    cdf_x = list(sorted([d[1] * 1000 for d in data]))
    cdf_y = np.arange(len(cdf_x)) / len(cdf_x)
    plt.close()
    plt.plot(cdf_x, cdf_y)
    plt.xlabel('Packet ACK delay (ms)')
    plt.ylabel('CDF')
    plt.ylim([0, 1])
    plt.xlim([0, max(cdf_x)])
    plt.savefig(os.path.join(OUTPUT_DIR, 'delay-cdf-packet.pdf'))


def analyze_network(context: StreamingContext) -> None:
    data = context.networking.pacing_rate_data
    x = [d[0] - context.start_ts for d in data]
    y = [d[1] / 1024 for d in data]
    yy = [d[2] / 1024 for d in data]
    plt.close()
    plt.plot(x, y)
    plt.plot(x, yy)
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Rate (Mbps)")
    plt.legend(["Pacing rate", "Padding rate"])
    plt.savefig(os.path.join(OUTPUT_DIR, 'pacing-rate.pdf'))
    ts_min = context.start_ts
    ts_max = max([p.sent_at for p in context.packets.values()])
    ts_range = ts_max - ts_min
    period = .3
    buckets = np.zeros(int(ts_range / period + 1))
    for p in context.packets.values():
        ts = (p.sent_at - ts_min)
        buckets[int(ts / period)] += p.size
    buckets = buckets / period * 8 / 1024 / 1024  # mbps
    plt.close()
    plt.plot(np.arange(len(buckets)) * period, buckets)
    plt.xlabel("Timestamp (s)")
    plt.ylabel("RTP egress rate (Mbps)")
    plt.savefig(os.path.join(OUTPUT_DIR, 'sending-rate.pdf'))


def print_statistics(context: StreamingContext) -> None:
    print("==========statistics==========")
    frame_ids = list(sorted(
        filter(lambda k: context.frames[k].codec != None, context.frames.keys())))
    frames_total = max(frame_ids) - min(frame_ids) + 1
    frames_recvd = len(frame_ids)
    print(
        f"Total frames: {frames_total}, loss rate: {(frames_total - frames_recvd) / frames_total:.2%}")


def analyze_codec(context: StreamingContext) -> None:
    plt.close()
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Timestamp (s)')
    ax1.set_ylabel('Bitrate (Kbps)', color='b')
    bitrate_data = np.array(context.bitrate_data)
    ax1.plot((bitrate_data[:, 0] - context.start_ts), bitrate_data[:, 1], 'b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.set_ylabel('FPS', color='r')
    fps_data = np.array(context.fps_data)
    ax2.plot((fps_data[:, 0] - context.start_ts), fps_data[:, 1], 'r')
    ax2.tick_params(axis='y', labelcolor='r')
    plt.savefig(os.path.join(OUTPUT_DIR, 'codec-params.pdf'))


def analyze_stream(context: StreamingContext) -> None:
    print_statistics(context)
    analyze_frame(context)
    analyze_packet(context)
    analyze_network(context)
    analyze_codec(context)


def main() -> None:
    log_dir = RESULTS_PATH
    sender_log = os.path.join(log_dir, 'sender.log')
    receiver_log = os.path.join(log_dir, 'receiver.log')
    context = StreamingContext()
    for line in open(sender_log).readlines():
        parse_line(line, context)
    analyze_stream(context)


if __name__ == "__main__":
    main()
