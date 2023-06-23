import os
import re
import matplotlib.pyplot as plt
import numpy as np

from pandia import RESULTS_PATH, DIAGRAMS_PATH

CODEC_NAMES = ['Generic', 'VP8', 'VP9', 'AV1', 'H264', 'Multiplex']
OUTPUT_DIR = DIAGRAMS_PATH

kDeltaTick=.25  # In ms
kBaseTimeTick=kDeltaTick*(1<<8)  # In ms
kTimeWrapPeriod=kBaseTimeTick * (1 << 24)  # In ms


class PacketContext(object):
    def __init__(self, rtp_id, payload_type, size, sent_at) -> None:
        self.sent_at = sent_at
        self.sent_at_utc = -1.0
        self.acked_at = -1
        self.received_at = -1  # Remote ts
        self.rtp_id = rtp_id
        self.payload_type = payload_type
        self.size = size
        self.received = None

    def ack_delay(self):
        return self.acked_at - self.sent_at if self.acked_at > 0 else -1

    def recv_delay(self):
        return self.received_at - self.sent_at_utc if self.received_at > 0 else -1


class FecContext(object):
    def __init__(self) -> None:
        self.fec_key_data = []
        self.fec_delta_data = []

class FrameContext(object):
    def __init__(self, frame_id, captured_at) -> None:
        self.frame_id = frame_id
        self.captured_at = captured_at
        self.captured_at_utc = 0
        self.width = 0
        self.height = 0
        self.encoded_at = 0
        self.assembled_at = 0
        self.assembled0_at = 0
        self.decoded_at = 0
        self.decoding_at = 0
        self.bitrate = 0
        self.codec = None
        self.encoded_size = 0
        self.encoded_shape = None
        self.rtp_packets: dict = {}
        self.rtp_id_range = [1000000, 0]
        self.rtp_packets_num = 0
        self.dropped_by_encoder = False
        self.is_key_frame = False
        self.qp = 0

    def last_rtp_send_ts(self):
        last_ts = -1
        for i in range(self.rtp_id_range[1], self.rtp_id_range[0] - 1, -1):
            pkt = self.rtp_packets.get(i, None)
            if pkt and pkt.sent_at > 0:
                return pkt.sent_at

    def last_rtp_recv_ts(self):
        last_ts = -1
        for i in range(self.rtp_id_range[1], self.rtp_id_range[0] - 1, -1):
            pkt = self.rtp_packets.get(i, None)
            if pkt and pkt.acked_at > 0:
                return pkt.acked_at
        return last_ts

    def encoding_delay(self):
        return self.encoded_at - self.captured_at if self.encoded_at > 0 else -1

    def assemble_delay(self):
        return self.assembled_at - self.captured_at if self.assembled_at > 0 else -1

    def pacing_delay(self):
        return self.last_rtp_send_ts() - self.captured_at if self.last_rtp_send_ts() else -1

    def decoding_queue_delay(self):
        return self.decoding_at - self.captured_at if self.decoding_at > 0 else -1

    def decoding_delay(self):
        return self.decoded_at - self.captured_at if self.decoded_at > 0 else -1

    def g2g_delay(self):
        return self.decoding_delay()

    def received(self):
        return self.assembled_at >= 0


class ActionContext(object):
    def __init__(self) -> None:
        self.bitrate = -1
        self.fps = -1
        self.resolution = -1
        self.pacing_rate = -1


class StreamingContext(object):
    def __init__(self) -> None:
        self.start_ts = 0
        self.frames: dict[int, FrameContext] = {}
        self.packets: dict[int, PacketContext] = {}
        self.packet_id_map = {}
        self.networking = NetworkContext()
        self.fec = FecContext()
        self.bitrate_data = []
        self.rtt_data = []
        self.fps_data = []
        self.codec_initiated = False
        self.last_captured_frame_id = 0
        self.last_decoded_frame_id = 0
        self.last_egress_packet_id = 0
        self.last_acked_packet_id = 0
        self.action_context = ActionContext()

    def reset_action_context(self):
        self.action_context = ActionContext()

    def codec(self) -> int:
        for i in range(self.last_captured_frame_id, -1, -1):
            frame = self.frames.get(i, None)
            if frame and frame.codec:
                return CODEC_NAMES.index(frame.codec)
        return 0

    def fps(self):
        res = 0
        for i in range(self.last_captured_frame_id, 0, -1):
            diff = self.frames[self.last_captured_frame_id].captured_at - \
                self.frames[i].captured_at
            if diff <= 1:
                if self.frames[i].encoded_at > 0 and self.frames[i].encoded_size > 0:
                    res += 1
            else:
                break
        return res

    def latest_frames(self, duration=1):
        res = []
        for i in range(self.last_captured_frame_id, 0, -1):
            if self.frames[i].captured_at >= self.frames[self.last_captured_frame_id].captured_at - duration:
                res.append(self.frames[i])
            else:
                break
        return res

    def latest_packets(self, duration=1):
        res = []
        for i in range(self.last_egress_packet_id, 0, -1):
            if self.packets[i].sent_at >= self.packets[self.last_egress_packet_id].sent_at - duration:
                res.append(self.packets[i])
            else:
                break
        return res

    def packet_delay(self, duration=1):
        res = []
        for i in range(self.last_egress_packet_id, 0, -1):
            pkt = self.packets[i]
            if pkt.sent_at >= self.packets[self.last_egress_packet_id].sent_at - duration:
                if pkt.recv_delay() >= 0:
                    res.append(pkt.recv_delay())
            else:
                break
        return np.mean(res) if res else 0

    def packet_delay_interval(self, duration=1):
        limit = 3
        buf = []
        res = []
        for i in range(self.last_egress_packet_id, 0, -1):
            pkt = self.packets[i]
            if pkt.sent_at >= self.packets[self.last_egress_packet_id].sent_at - duration:
                if pkt.received_at > 0:
                    if buf:
                        a = pkt.received_at - buf[0].received_at
                        b = pkt.sent_at - buf[0].sent_at
                        if b != 0:
                            res.append(a / b) 
                    buf.append(pkt)
                    if len(buf) > limit:
                        buf.pop(0)
            else:
                break
        return np.mean(res) if res else 0

    def packet_rtt_measured(self, duration=1):
        res = []
        for i in range(len(self.rtt_data) - 1, -1, -1):
            if self.rtt_data[i][0] >= self.rtt_data[-1][0] - duration:
                res.append(self.rtt_data[i][1])
            else:
                break
        if res:
            return np.mean(res)
        if self.rtt_data:
            return self.rtt_data[-1][1]
        return 0


    def packet_loss_rate(self, duration=1):
        # TODO: We currently do not count recently lost packets
        sent_count = 0
        received_count = 0
        for i in range(self.last_egress_packet_id, 0, -1):
            if self.packets[i].sent_at >= self.packets[self.last_egress_packet_id].sent_at - duration:
                if self.packets[i].received_at > 0:
                    received_count += 1
                if received_count > 0:
                    sent_count += 1
            else:
                break
        return received_count / sent_count

    def latest_egress_packets(self, duration=1):
        res = []
        for i in range(self.last_egress_packet_id, 0, -1):
            if i in self.packets and \
                self.packets[i].sent_at >= self.packets[self.last_egress_packet_id].sent_at - duration:
                res.append(self.packets[i])
            else:
                break
        return res

    def latest_acked_packets(self, duration=1):
        res = []
        for i in range(self.last_acked_packet_id, 0, -1):
            if i in self.packets and \
                self.packets[i].acked_at >= self.packets[self.last_acked_packet_id].acked_at - duration:
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
    if (line.startswith('(video_capture_impl.cc') or line.startswith('(frame_generator_capturer')) and 'FrameCaptured' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] FrameCaptured, id: (\\d+), width: (\\d+), height: (\\d+), ts: (\\d+), utc ts: (\\d+) ms.*'), line)
        ts = int(m[1]) / 1000
        frame_id = int(m[2])
        width = int(m[3])
        height = int(m[4])
        utc_ts = int(m[6])
        frame = FrameContext(frame_id, ts)
        frame.captured_at_utc = utc_ts
        context.last_captured_frame_id = frame_id
        context.frames[frame_id] = frame
    elif line.startswith('(video_codec_initializer.cc') and 'SetupCodec' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] SetupCodec.*'), line)
        ts = int(m[1]) / 1000
        context.codec_initiated = True
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
        frame: FrameContext = context.frames.get(frame_id, None)
        if frame:
            frame.rtp_packets[rtp_id] = None
            frame.rtp_id_range[0] = min(frame.rtp_id_range[0], rtp_id)
            frame.rtp_id_range[1] = max(frame.rtp_id_range[1], rtp_id)
    elif line.startswith('(video_stream_encoder.cc') and 'Start encoding' in line:
        m = re.match(re.compile(
            '.*Start encoding, id: (\\d+), frame shape: (\\d+)x(\\d+).*'), line)
        frame_id = int(m[1])
        width = int(m[2])
        height = int(m[3])
        context.action_context.resolution = width
        frame: FrameContext = context.frames.get(frame_id, None)
        if frame:
            frame.width = width
            frame.height = height
    elif (line.startswith('(h264_encoder_impl.cc') or line.startswith('(libvpx_vp8_encoder.cc')) and 'Start encoding' in line:
        m = re.match(re.compile(
            '.*Start encoding, frame id: (\\d+), bitrate: (\\d+) kbps.*'), line)
        frame_id = int(m[1])
        bitrate = int(m[2])
        if context.action_context.bitrate <= 0:
            context.action_context.bitrate = bitrate
        frame: FrameContext = context.frames.get(frame_id, None)
        if frame:
            frame: FrameContext = context.frames[frame_id]
            frame.bitrate = bitrate
    elif (line.startswith('(h264_encoder_impl.cc') or line.startswith('(libvpx_vp8_encoder.cc')) and 'Finish encoding' in line:
        m = re.match(re.compile(
            '.*Finish encoding, frame id: (\\d+), frame type: (\\d+), frame size: (\\d+), qp: (\\d+).*'), line)
        frame_id = int(m[1])
        frame_type = int(m[2])
        frame_size = int(m[3])
        qp = int(m[4])
        frame: FrameContext = context.frames.get(frame_id, None)
        if frame:
            frame: FrameContext = context.frames[frame_id]
            frame.is_key_frame = frame_type == 3
            frame.dropped_by_encoder = frame_size == 0
            frame.qp = qp
    elif line.startswith('(rtp_rtcp_impl2.cc') and 'Assign sequence number' in line:
        m = re.match(re.compile(
            '.*Assign sequence number, id: (\\d+), sequence number: (\\d+).*'), line)
        rtp_id = int(m[1])
        sequence_number = int(m[2])
        context.packet_id_map[sequence_number] = rtp_id
    elif line.startswith('(rtcp_receiver.cc') and 'RTCP RTT' in line:
        m = re.match(re.compile('.*\\[(\\d+)\\] RTCP RTT: (\\d+) ms.*'), line)
        ts = int(m[1]) / 1000
        rtt = int(m[2]) / 1000
        context.rtt_data.append((ts, rtt))
    elif line.startswith('(rtp_transport_controller_send.cc') and 'OnSentPacket' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] OnSentPacket, id: (-?\\d+), type: (\\d+), size: (\\d+), utc: (\\d+) ms.*'), line)
        ts = int(m[1]) / 1000
        rtp_id = int(m[2])
        rtp_type = int(m[3])
        size = int(m[4])
        utc = int(m[5]) / 1000
        if rtp_id >= 0:
            packet = PacketContext(rtp_id, rtp_type, size, ts)
            packet.sent_at_utc = utc
            if packet.payload_type == 1:
                context.packets[packet.rtp_id] = packet
                for i in range(context.last_captured_frame_id, 0, -1):
                    frame = context.frames[i]
                    if rtp_id in frame.rtp_packets:
                        frame.rtp_packets[rtp_id] = packet
            context.last_egress_packet_id = max(
                rtp_id, context.last_egress_packet_id)
    elif line.startswith('(transport_feedback.cc:') and 'RTCP feedback, packet acked' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] RTCP feedback, packet acked: (\\d+) at (\\d+) ms.*'), line)
        ts = int(m[1]) / 1000
        rtp_id = int(m[2])
        # The recv time is wrapped by kTimeWrapPeriod. 
        # The fixed value 1570 should be calculated according to the current time.
        received_at = (int(m[3]) + 1570 * kTimeWrapPeriod) / 1000
        packet = context.packets.get(rtp_id, None)
        if packet:
            packet.received_at = received_at
    elif line.startswith('(transport_feedback_demuxer.cc:') and 'Packet acked' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] Packet acked, id: (\\d+), received: (\\d+), delta_sum: (-?\\d+).*'), line)
        ts = int(m[1]) / 1000
        rtp_id = int(m[2])
        received = int(m[3])
        delta = int(m[4]) / 1000
        packet: PacketContext = context.packets.get(rtp_id, None)
        if packet:
            packet.acked_at = ts - delta
            packet.received = received == 1
            context.last_acked_packet_id = max(
                rtp_id, context.last_acked_packet_id)
    elif line.startswith('(ulpfec_generator.cc:') and 'SetProtectionParameters' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] SetProtectionParameters, delta: (\\d+), key: (\\d+).*'), line)
        ts = int(m[1]) / 1000
        fec_delta = int(m[2])
        fec_key = int(m[3])
        context.fec.fec_key_data.append((ts, fec_key))
        context.fec.fec_delta_data.append((ts, fec_delta))
    elif line.startswith('(rtp_sender_video.cc:') and 'SendVideo' in line:
        m = re.match(re.compile(
            '.*SendVideo, frame id: (\\d+), number of RTP packets: (\\d+).*'), line)
        frame_id = int(m[1])
        rtp_packets = int(m[2])
        frame = context.frames.get(frame_id, None)
        if frame:
            frame.rtp_packets_num = rtp_packets
    elif line.startswith('(rtcp_receiver.cc') and 'Frame decoding acked' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] Frame decoding acked, id: (\\d+), receiving offset: (\\d+), decoding offset: (\\d+).*'), line)
        ts = int(m[1]) / 1000
        rtp_sequence = int(m[2])
        recving_offset = int(m[3]) / 1000
        decoding_offset = int(m[4]) / 1000
        rtp_id = context.packet_id_map.get(rtp_sequence, None)
        if rtp_id:
            for i in range(context.last_captured_frame_id, 0, -1):
                frame: FrameContext = context.frames.get(i, None)
                if frame:
                    if frame.rtp_id_range[0] == rtp_id:
                        frame.decoded_at = ts
                        frame.decoding_at = ts - decoding_offset
                        frame.assembled0_at = ts - recving_offset
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
            frame: FrameContext = context.frames.get(i, None)
            if frame:
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
        context.action_context.pacing_rate = pacing_rate
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


def analyze_frame(context: StreamingContext, output_dir=OUTPUT_DIR) -> None:
    frame_id_list = list(sorted(context.frames.keys()))
    data_frame_delay = []
    lost_frames = []
    key_frames = []
    started = False
    last_frame_id = -1
    for frame_id in frame_id_list:
        frame: FrameContext = context.frames[frame_id]
        if frame.is_key_frame:
            key_frames.append((frame.captured_at - context.start_ts, frame.encoded_size / 1024))
        if frame.codec and frame.encoded_at > 0:
            started = True
            last_frame_id = frame_id
            data_frame_delay.append({'id': frame_id,
                                     'ts': frame.captured_at,
                                     'encoded by': frame.encoding_delay(),
                                     'paced by': frame.pacing_delay(),
                                     'assembled by': frame.assemble_delay(),
                                     'queued by': frame.decoding_queue_delay(),
                                     'decoded by': frame.decoding_delay(),})
        elif started:
            lost_frames.append(frame.captured_at - context.start_ts)
    plt.close()
    ylim = 0
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, k in enumerate(['decoded by', 'queued by', 'assembled by', 'paced by', 'encoded by'][::-1]):
        x = np.array([d['ts'] - context.start_ts for d in data_frame_delay])
        y = np.array([d[k] for d in data_frame_delay]) * 1000
        print(f'Median: {k} {np.median(y)} ms')
        ylim = np.percentile(y, 50)
        indexes = (y > 0).nonzero()
        plt.plot(x[indexes], y[indexes], colors[i])
    plt.plot(lost_frames, [10 for _ in lost_frames], 'x')
    plt.legend(['Decoding', 'Queue', 'Transmission', 'Pacing', 'Encoding'][::-1] + ['Lost'])
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Delay (ms)')
    plt.ylim([0, ylim * 1.8])
    plt.savefig(os.path.join(output_dir, 'delay-frame.pdf'))
    x = []
    y = []
    bitrates = []
    for frame_id in frame_id_list:
        frame: FrameContext = context.frames[frame_id]
        if (frame.encoded_size):
            x.append(frame.captured_at - context.start_ts)
            y.append(frame.encoded_size / 1024)
            bitrates.append(frame.bitrate)
    qp_data = [(context.frames[frame_id].captured_at - context.start_ts, context.frames[frame_id].qp)
               for frame_id in frame_id_list if context.frames[frame_id].encoded_size > 0]
    plt.close()
    fig, ax1 = plt.subplots()
    ax1.plot(x, y)
    ax1.plot(lost_frames, [10 for _ in lost_frames], 'x')
    ax1.plot([f[0] for f in key_frames], [f[1] for f in key_frames], 'o')
    ax1.set_xlabel('Timestamp (s)')
    ax1.set_ylabel('Encoded size (KB)')
    ax1.legend(['Encoded size', 'Lost frames', 'Key frames'])
    ax2 = ax1.twinx()
    ax2.plot(x, bitrates, 'r')
    ax2.set_ylabel('Bitrate (Kbps)')
    plt.savefig(os.path.join(output_dir, 'size-frame.pdf'))

    plt.close()
    plt.plot([f[0] for f in qp_data], [f[1] for f in qp_data])
    plt.xlabel('Timestamp (s)')
    plt.ylabel('QP')
    plt.savefig(os.path.join(output_dir, 'qp-frame.pdf'))

    plt.close()
    frames = filter(lambda x: x.encoded_at > 0, context.frames.values())
    ts_list = [f.encoded_at for f in frames]
    ts_list = sorted(ts_list)
    duration = 1
    bucks = (ts_list[-1] - ts_list[0]) / duration + 1
    data = np.zeros((int(bucks), 1))
    for t in ts_list:
        data[int((t - ts_list[0]) / duration)] += 1
    x = [i * duration for i in range(len(data))]
    plt.plot(x, data)
    plt.xlabel('Timestamp (s)')
    plt.ylabel('FPS')
    plt.savefig(os.path.join(output_dir, 'fps.pdf'))


def analyze_packet(context: StreamingContext, output_dir=OUTPUT_DIR) -> None:
    data_ack = []
    data_recv = []
    for pkt in sorted(context.packets.values(), key=lambda x: x.sent_at):
        pkt: PacketContext = pkt
        if pkt.ack_delay() > 0:
            data_ack.append((pkt.sent_at, pkt.ack_delay()))
        if pkt.recv_delay() != -1:
            data_recv.append((pkt.sent_at, pkt.recv_delay()))
    plt.close()
    x = [(d[0] - context.start_ts) for d in data_recv]
    y = [d[1] * 1000 for d in data_recv]
    plt.plot(x, y, 'x')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('RTT (ms)')
    if y:
        plt.ylim([min(y), max(y)])
    plt.savefig(os.path.join(output_dir, 'delay-packet-biased.pdf'))

    plt.close()
    plt.plot([(d[0] - context.start_ts)
             for d in data_ack], [d[1] * 1000 for d in data_ack], 'x')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('RTT (ms)')
    # plt.ylim([0, 50])
    plt.savefig(os.path.join(output_dir, 'delay-packet.pdf'))

    cdf_x = list(sorted([d[1] * 1000 for d in data_ack]))
    cdf_y = np.arange(len(cdf_x)) / len(cdf_x)
    plt.close()
    plt.plot(cdf_x, cdf_y)
    plt.xlabel('Packet ACK delay (ms)')
    plt.ylabel('CDF')
    plt.ylim([0, 1])
    plt.xlim([0, max(cdf_x)])
    plt.savefig(os.path.join(output_dir, 'delay-packet-cdf.pdf'))


def analyze_network(context: StreamingContext, output_dir=OUTPUT_DIR) -> None:
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
    plt.savefig(os.path.join(output_dir, 'pacing-rate.pdf'))
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
    plt.savefig(os.path.join(output_dir, 'sending-rate.pdf'))

    plt.close()
    x = [r[0] for r in context.rtt_data]
    y = [r[1] * 1000 for r in context.rtt_data]
    plt.plot(x, y)
    plt.xlabel("Timestamp (s)")
    plt.ylabel("RTT (ms)")
    plt.savefig(os.path.join(output_dir, 'rtt.pdf'))


def print_statistics(context: StreamingContext) -> None:
    print("==========statistics==========")
    frame_ids = list(sorted(
        filter(lambda k: context.frames[k].codec != None, context.frames.keys())))
    frames_total = max(frame_ids) - min(frame_ids) + 1
    frames_recvd = len(frame_ids)
    print(
        f"Total frames: {frames_total}, loss rate: {(frames_total - frames_recvd) / frames_total:.2%}")


def analyze_codec(context: StreamingContext, output_dir=OUTPUT_DIR) -> None:
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
    plt.savefig(os.path.join(output_dir, 'codec-params.pdf'))

def analyze_fec(context: StreamingContext, output_dir=OUTPUT_DIR) -> None:
    plt.close()
    plt.plot([d[0] - context.start_ts for d in context.fec.fec_key_data],
             [d[1] for d in context.fec.fec_key_data], '--')
    plt.plot([d[0] - context.start_ts for d in context.fec.fec_delta_data],
             [d[1] for d in context.fec.fec_delta_data], '-.')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('FEC Ratio')
    plt.legend(['Key frame FEC', 'Delta frame FEC'])
    plt.savefig(os.path.join(output_dir, 'fec.pdf'))


def analyze_stream(context: StreamingContext, output_dir=OUTPUT_DIR) -> None:
    print_statistics(context)
    analyze_frame(context, output_dir)
    analyze_packet(context, output_dir)
    analyze_network(context, output_dir)
    analyze_codec(context, output_dir)
    analyze_fec(context, output_dir)


def main() -> None:
    # sender_log = '/tmp/eval_sender_log.txt'
    sender_log = os.path.join(DIAGRAMS_PATH, 'eval_rllib', 'eval_sender_log.txt')
    context = StreamingContext()
    for line in open(sender_log).readlines():
        parse_line(line, context)
    analyze_stream(context)


if __name__ == "__main__":
    main()
