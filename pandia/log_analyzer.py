import os
import re
import time
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
        self.retrans_ref = None
        self.rtx = False
        self.received = None  # The reception is reported by RTCP transport feedback, which is biased because the feedback may be lost.
        self.nack_retransmitted = False 

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
        self.encoding_at = .0
        self.encoded_at = .0
        self.assembled_at = .0
        self.assembled0_at = .0
        self.decoded_at = .0
        self.decoding_at = .0
        self.bitrate = 0
        self.codec = 0
        self.encoded_size = 0
        self.encoded_shape = (0, 0)
        self.rtp_packets: dict = {}
        self.rtp_id_range = [1000000, 0]
        self.dropped_by_encoder = False
        self.is_key_frame = False
        self.qp = 0
        self.retrans_record = {}

    def last_rtp_send_ts_including_rtx(self):
        last_retrans_list = [self.rtp_packets[r[-1]] for r in self.retrans_record.values()]
        if last_retrans_list:
            return max([l.sent_at for l in last_retrans_list])
        return self.last_rtp_send_ts()


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

    def pacing_delay_including_rtx(self):
        return self.last_rtp_send_ts_including_rtx() - self.captured_at if self.last_rtp_send_ts_including_rtx() else -1

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
        self.packet_loss_data = []
        self.fps_data = []
        self.codec_initiated = False
        self.last_captured_frame_id = 0
        self.last_decoded_frame_id = 0
        self.last_egress_packet_id = 0
        self.last_acked_packet_id = 0
        self.action_context = ActionContext()
        self.packet_frame_map = {}
        self.padding_rtps = set()
        self.packet_retrans_map = {}

    def reset_action_context(self):
        self.action_context = ActionContext()

    def codec_name(self) -> int:
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
    if 'FrameCaptured' in line:
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
    elif 'UpdateFecRates' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] UpdateFecRates, fraction lost: ([0-9.]+).*'), line)
        ts = int(m[1]) / 1000
        loss_rate = float(m[2])
        context.packet_loss_data.append((ts, loss_rate))
    elif 'SetupCodec' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] SetupCodec.*'), line)
        ts = int(m[1]) / 1000
        context.codec_initiated = True
        context.start_ts = ts
    elif 'Frame encoded' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] Frame encoded, id: (\\d+), codec: (\\d+), size: (\\d+), width: (\\d+), height: (\\d+), .*'), line)
        ts = int(m[1]) / 1000
        frame_id = int(m[2])
        codec = int(m[3])
        encoded_size = int(m[4])
        width = int(m[5])
        height = int(m[6])
        frame: FrameContext = context.frames[frame_id]
        frame.encoded_at = ts
        frame.codec = codec
        frame.encoded_shape = (width, height)
        frame.encoded_size = encoded_size
    elif 'Assign RTP id' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] .*Assign RTP id, id: (\\d+), frame id: (\\d+), type: (\\d+), retrans seq num: (\\d+), allow retrans: (\\d+).*'), line)
        ts = int(m[1]) / 1000
        rtp_id = int(m[2])
        frame_id = int(m[3])
        rtp_type = int(m[4])  # 0: audio, 1: video, 2: rtx, 3: fec, 4: padding
        retrans_seq_num = int(m[5])
        allow_retrans = int(m[6]) == 1
        context.packet_frame_map[rtp_id] = frame_id
        if rtp_type == 4:
            context.padding_rtps.add(rtp_id)
        elif frame_id > 0 and frame_id in context.frames:
            frame: FrameContext = context.frames[frame_id]
            frame.rtp_packets[rtp_id] = None
            if rtp_type == 2:
                original_rtp_id = context.packet_id_map[retrans_seq_num]
                frame.retrans_record.setdefault(original_rtp_id, []).append(rtp_id)
                context.packet_retrans_map[rtp_id] = original_rtp_id
            elif rtp_type == 1:
                frame.rtp_id_range[0] = min(frame.rtp_id_range[0], rtp_id)
                frame.rtp_id_range[1] = max(frame.rtp_id_range[1], rtp_id)
    elif 'NVENC Start encoding' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] NVENC Start encoding, frame id: (\\d+), shape: (\\d+) x (\\d+), bitrate: (\\d+) kbps.*'), line)
        ts = int(m[1]) / 1000
        frame_id = int(m[2])
        width = int(m[3])
        height = int(m[4])
        bitrate = int(m[5])
        context.action_context.resolution = width
        if context.action_context.bitrate <= 0:
            context.action_context.bitrate = bitrate
        if frame_id in context.frames:
            frame: FrameContext = context.frames[frame_id]
            frame.bitrate = bitrate
            frame.width = width
            frame.height = height
            frame.encoding_at = ts
    elif 'Finish encoding' in line:
        m = re.match(re.compile(
            '.*Finish encoding, frame id: (\\d+), frame type: (\\d+), frame size: (\\d+), is key: (\\d+), qp: (-?\\d+).*'), line)
        frame_id = int(m[1])
        frame_type = int(m[2])
        frame_size = int(m[3])
        is_key = int(m[4])
        qp = int(m[5])
        if frame_id in context.frames:
            frame = context.frames[frame_id]
            frame.qp = qp
            frame.is_key_frame = is_key != 0
    elif 'Assign sequence number' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] Assign sequence number, id: (\\d+), sequence number: (\\d+).*'), line)
        ts = int(m[1]) / 1000
        rtp_id = int(m[2])
        sequence_number = int(m[3])
        context.packet_id_map[sequence_number] = rtp_id
    elif 'RTCP RTT' in line:
        m = re.match(re.compile('.*\\[(\\d+)\\] RTCP RTT: (\\d+) ms.*'), line)
        ts = int(m[1]) / 1000
        rtt = int(m[2]) / 1000
        context.rtt_data.append((ts, rtt))
    elif 'ReSendPacket' in line:
        m = re.match(re.compile('.*\\[(\\d+)\\] ReSendPacket, id: (\\d+).*'), line)
        ts = int(m[1]) / 1000
        sequence_number = int(m[2])
        if sequence_number in context.packet_id_map:
            rtp_id = context.packet_id_map[sequence_number]
            packet = context.packets[rtp_id]
            packet.nack_retransmitted = True
    elif 'OnSentPacket' in line:
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
                frame_id = context.packet_frame_map[packet.rtp_id]
                if frame_id in context.frames:
                    frame: FrameContext = context.frames[frame_id]
                    if rtp_id in frame.rtp_packets:
                        frame.rtp_packets[rtp_id] = packet
            context.last_egress_packet_id = max(
                rtp_id, context.last_egress_packet_id)
    elif 'RTCP feedback, packet acked' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] RTCP feedback, packet acked: (\\d+) at (\\d+) ms.*'), line)
        ts = int(m[1]) / 1000
        rtp_id = int(m[2])
        # The recv time is wrapped by kTimeWrapPeriod.
        # The fixed value 1570 should be calculated according to the current time.
        offset = int(time.time() * 1000 / kTimeWrapPeriod) - 1
        received_at = (int(m[3]) + offset * kTimeWrapPeriod) / 1000
        packet = context.packets.get(rtp_id, None)
        if packet:
            packet.received_at = received_at
    elif 'Packet acked' in line:
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
    elif 'SetProtectionParameters' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] SetProtectionParameters, delta: (\\d+), key: (\\d+).*'), line)
        ts = int(m[1]) / 1000
        fec_delta = int(m[2])
        fec_key = int(m[3])
        context.fec.fec_key_data.append((ts, fec_key))
        context.fec.fec_delta_data.append((ts, fec_delta))
    elif 'Frame decoding acked' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] Frame decoding acked, id: (\\d+), receiving offset: (\\d+), decoding offset: (\\d+).*'), line)
        ts = int(m[1]) / 1000
        rtp_sequence = int(m[2])
        recving_offset = int(m[3]) / 1000
        decoding_offset = int(m[4]) / 1000
        rtp_id = context.packet_id_map.get(rtp_sequence, None)
        if rtp_id:
            for i in range(context.last_captured_frame_id, 0, -1):
                if i in context.frames:
                    frame: FrameContext = context.frames[i]
                    if frame.rtp_id_range[0] == rtp_id:
                        frame.decoded_at = ts
                        frame.decoding_at = ts - decoding_offset
                        frame.assembled0_at = ts - recving_offset
                        context.last_decoded_frame_id = max(
                            frame.frame_id, context.last_decoded_frame_id)
                        break
                    if frame.rtp_id_range[0] < rtp_id:
                        break
    elif 'Frame reception acked' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] Frame reception acked, id: (\\d+).*'), line)
        ts = int(m[1]) / 1000
        rtp_sequence = int(m[2])
        rtp_id = context.packet_id_map[rtp_sequence]
        for i in range(context.last_captured_frame_id, 0, -1):
            if i in context.frames:
                frame: FrameContext = context.frames[i]
                if frame.rtp_id_range[0] == rtp_id:
                    frame.assembled_at = ts
                    break
                if frame.rtp_id_range[0] < rtp_id:
                    break
    elif 'SetPacingRates' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] SetPacingRates, pacing rate: (\\d+) kbps, pading rate: (\\d+) kbps.*'), line)
        ts = int(m[1]) / 1000
        pacing_rate = int(m[2])
        padding_rate = int(m[3])
        context.action_context.pacing_rate = pacing_rate
        context.networking.pacing_rate_data.append(
            [ts, pacing_rate, padding_rate])
    elif 'SetRates, ' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] SetRates, stream id: (\\d+), bitrate: (\\d+) kbps, '
            'max bitrate: (\\d+) kbps, framerate: (\\d+).*'), line)
        ts = int(m[1]) / 1000
        stream_id = int(m[2]) 
        bitrate = int(m[3])
        bitrate_max = int(m[4])
        fps = int(m[5])
        context.bitrate_data.append([ts, bitrate, bitrate_max])
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
                                     'paced by (rtx)': frame.pacing_delay_including_rtx(),
                                     'assembled by': frame.assemble_delay(),
                                     'queued by': frame.decoding_queue_delay(),
                                     'decoded by': frame.decoding_delay(),})
        elif started:
            lost_frames.append(frame.captured_at - context.start_ts)
    plt.close()
    ylim = 0
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, k in enumerate(['decoded by', 'queued by', 'assembled by', 'paced by (rtx)', 'paced by', 'encoded by']):
        x = np.array([d['ts'] - context.start_ts for d in data_frame_delay])
        y = np.array([d[k] for d in data_frame_delay]) * 1000
        print(f'Median: {k} {np.median(y)} ms')
        if ylim == 0:
            ylim = np.percentile(y, 90)
        indexes = (y > 0).nonzero()
        plt.plot(x[indexes], y[indexes], colors[i])
    # plt.plot(lost_frames, [10 for _ in lost_frames], 'x')
    plt.legend(['Decoding', 'Queue', 'Transmission', 'Pacing (RTX)', 'Pacing', 'Encoding'])
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Delay (ms)')
    plt.ylim([0, ylim])
    plt.savefig(os.path.join(output_dir, 'mea-delay-frame.pdf'))
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
    plt.savefig(os.path.join(output_dir, 'mea-size-frame.pdf'))

    plt.close()
    x = []
    y = []
    yy = []
    for frame in context.frames.values():
        count_all = 0
        count_lost = len(frame.retrans_record)
        x.append(frame.captured_at - context.start_ts)
        for i in range(frame.rtp_id_range[0], frame.rtp_id_range[1] + 1):
            if i in context.padding_rtps or i in context.packet_retrans_map:
                continue
            count_all += 1
        yy.append(count_lost)
        y.append(count_lost / count_all * 100 if count_all > 0 else 0)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, y, 'b')
    x = np.array(x)
    yy = np.array(yy)
    # ax2.plot(x[yy > 0], yy[yy > 0], 'r.')
    ax2.plot(x, yy, 'r.')
    plt.xlabel('Timestamp (s)')
    ax1.set_ylabel('Packet loss rate per frame (%)')
    ax2.set_ylabel('Number of lost packets per frame')
    plt.savefig(os.path.join(output_dir, 'mea-loss-packet-frame.pdf'))

    plt.close()
    fig, ax1 = plt.subplots()
    duration = 1
    start = 0
    count = y[0]
    accu = [count]
    fps = [1]
    for i in range(1, len(x)):
        while start < i and x[i] - x[start] > duration:
            start += 1
            count -= y[start]
        count += y[i]
        fps.append(i - start + 1)
        accu.append(count)
    ax1.plot(x, accu)
    ax1.set_xlabel('Timestamp (s)')
    ax1.set_ylabel('Accumulated frame size (KB)')
    ax2 = ax1.twinx()
    ax2.plot(x, fps, 'r')
    ax2.set_ylabel('FPS')
    plt.savefig(os.path.join(output_dir, 'mea-size-frame-accu.pdf'))

    plt.close()
    accu = np.array(accu)
    plt.plot(x, accu * 8 / duration)
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Bitrate (Kbps)')
    plt.savefig(os.path.join(output_dir, 'mea-bitrate.pdf'))

    plt.close()
    plt.plot([f[0] for f in qp_data], [f[1] for f in qp_data])
    plt.xlabel('Timestamp (s)')
    plt.ylabel('QP')
    plt.savefig(os.path.join(output_dir, 'rep-qp-frame.pdf'))

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
    plt.savefig(os.path.join(output_dir, 'mea-fps.pdf'))


def analyze_packet(context: StreamingContext, output_dir=OUTPUT_DIR) -> None:
    data_ack = []
    data_recv = []
    offset = -0.00335956
    for pkt in sorted(context.packets.values(), key=lambda x: x.sent_at):
        pkt: PacketContext = pkt
        if pkt.ack_delay() > 0:
            data_ack.append((pkt.sent_at, pkt.ack_delay()))
        if pkt.recv_delay() != -1:
            data_recv.append((pkt.sent_at, pkt.recv_delay() - offset))
    plt.close()
    x = [(d[0] - context.start_ts) for d in data_recv]
    y = [d[1] * 1000 for d in data_recv]
    plt.plot(x, y, '.')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Packet transmission delay (ms)')
    if y:
        plt.ylim([min(y), max(y)])
    plt.ylim([0, 10])
    plt.savefig(os.path.join(output_dir, 'mea-delay-packet-biased.pdf'))

    plt.close()
    plt.plot([(d[0] - context.start_ts)
             for d in data_ack], [d[1] * 1000 for d in data_ack], 'x')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('RTT (ms)')
    # plt.ylim([0, 50])
    plt.savefig(os.path.join(output_dir, 'mea-delay-packet.pdf'))

    cdf_x = list(sorted([d[1] * 1000 for d in data_ack]))
    cdf_y = np.arange(len(cdf_x)) / len(cdf_x)
    plt.close()
    plt.plot(cdf_x, cdf_y)
    plt.xlabel('Packet ACK delay (ms)')
    plt.ylabel('CDF')
    plt.ylim([0, 1])
    plt.xlim([0, max(cdf_x)])
    plt.savefig(os.path.join(output_dir, 'mea-delay-packet-cdf.pdf'))

    duration = 1
    packets = sorted(context.packets.values(), key=lambda x: x.sent_at)
    bucks_sent = np.zeros(int((packets[-1].sent_at - packets[0].sent_at) / duration + 1))
    bucks_lost = np.zeros(int((packets[-1].sent_at - packets[0].sent_at) / duration + 1))
    for p in packets:
        i = int((p.sent_at - packets[0].sent_at) / duration)
        if p.rtp_id >= 0 and p.received is not None:
            bucks_sent[i] += 1
            if not p.received:
                bucks_lost[i] += 1
    x = [i * duration for i in range(len(bucks_sent))]
    y = bucks_lost / bucks_sent * 100
    plt.close()
    plt.plot(x, y)
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Packet loss rate (%)')
    plt.savefig(os.path.join(output_dir, 'mea-loss-packet.pdf'))

    duration = 1
    packets = sorted(context.packets.values(), key=lambda x: x.sent_at)
    bucks_sent = np.zeros(int((packets[-1].sent_at - packets[0].sent_at) / duration + 1))
    bucks_retrans = np.zeros(int((packets[-1].sent_at - packets[0].sent_at) / duration + 1))
    for p in packets:
        i = int((p.sent_at - packets[0].sent_at) / duration)
        if p.rtp_id >= 0:
            bucks_sent[i] += 1
            if p.nack_retransmitted:
                bucks_retrans[i] += 1
    x = [i * duration for i in range(len(bucks_sent))]
    y = bucks_retrans / bucks_sent * 100
    plt.close()
    plt.plot(x, y)
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Packet retransmission ratio (%)')
    plt.savefig(os.path.join(output_dir, 'mea-retrans-packet.pdf'))

    plt.close()
    x = [i[0] - context.start_ts for i in context.packet_loss_data]
    y = [i[1] * 100 for i in context.packet_loss_data]
    plt.plot(x, y)
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Packet loss rate (%)')
    plt.savefig(os.path.join(output_dir, 'rep-loss-packet.pdf'))


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
    plt.savefig(os.path.join(output_dir, 'set-pacing-rate.pdf'))
    ts_min = context.start_ts
    ts_max = max([p.sent_at for p in context.packets.values()])
    ts_range = ts_max - ts_min
    period = .005
    buckets = np.zeros(int(ts_range / period + 1))
    for p in context.packets.values():
        ts = (p.sent_at - ts_min)
        if ts >= 0:
            buckets[int(ts / period)] += p.size
    buckets = buckets / period * 8 / 1024 / 1024  # mbps
    plt.close()
    plt.plot(np.arange(len(buckets)) * period, buckets, '.')
    plt.xlabel("Timestamp (s)")
    plt.ylabel("RTP egress rate (Mbps)")
    plt.savefig(os.path.join(output_dir, 'mea-sending-rate.pdf'))

    plt.close()
    x = [r[0] for r in context.rtt_data]
    y = [r[1] * 1000 for r in context.rtt_data]
    plt.plot(x, y)
    plt.xlabel("Timestamp (s)")
    plt.ylabel("RTT (ms)")
    plt.savefig(os.path.join(output_dir, 'rep-rtt.pdf'))


def print_statistics(context: StreamingContext) -> None:
    print("========== STATISTICS [SENDER] ==========")
    frame_ids = list(filter(lambda k: context.frames[k].codec, context.frames.keys()))
    id_min = min(frame_ids) if frame_ids else 0
    id_max = max(frame_ids) if frame_ids else 0
    frames_total = (id_max - id_min + 1) if frame_ids else 0
    frames_sent = len(frame_ids)
    frame_ids = list(filter(lambda f: id_min <= f.frame_id <= id_max and f.decoded_at > 0, context.frames.values()))
    frames_decoded = len(frame_ids)
    loss_rate_encoding = ((frames_total - frames_sent) / frames_total) if frames_total else 0
    loss_rate_decoding = ((frames_total - frames_decoded) / frames_total) if frames_total else 0
    print(f"Total frames: {frames_total}, encoding loss rate: {loss_rate_encoding:.2%}, "
          f"decoding loss rate: {loss_rate_decoding:.2%}")


def analyze_codec(context: StreamingContext, output_dir=OUTPUT_DIR) -> None:
    plt.close()
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Timestamp (s)')
    ax1.set_ylabel('Bitrate (Kbps)', color='b')
    bitrate_data = np.array(context.bitrate_data)
    ax1.plot((bitrate_data[:, 0] - context.start_ts), bitrate_data[:, 1], 'b')
    ax1.plot((bitrate_data[:, 0] - context.start_ts), bitrate_data[:, 2], 'b-.')
    ax1.legend(['Target', 'Max'])
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.set_ylabel('FPS', color='r')
    fps_data = np.array(context.fps_data)
    ax2.plot((fps_data[:, 0] - context.start_ts), fps_data[:, 1], 'r')
    ax2.tick_params(axis='y', labelcolor='r')
    plt.savefig(os.path.join(output_dir, 'set-codec-params.pdf'))

def analyze_fec(context: StreamingContext, output_dir=OUTPUT_DIR) -> None:
    plt.close()
    plt.plot([d[0] - context.start_ts for d in context.fec.fec_key_data],
             [d[1] for d in context.fec.fec_key_data], '--')
    plt.plot([d[0] - context.start_ts for d in context.fec.fec_delta_data],
             [d[1] for d in context.fec.fec_delta_data], '-.')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('FEC Ratio')
    plt.legend(['Key frame FEC', 'Delta frame FEC'])
    plt.savefig(os.path.join(output_dir, 'set-fec.pdf'))


def analyze_stream(context: StreamingContext, output_dir=OUTPUT_DIR) -> None:
    print_statistics(context)
    analyze_frame(context, output_dir)
    analyze_packet(context, output_dir)
    analyze_network(context, output_dir)
    analyze_codec(context, output_dir)
    analyze_fec(context, output_dir)


def main() -> None:
    # sender_log = '/tmp/eval_sender_log.txt'
    # sender_log = '/tmp/test_sender.log'
    working_dir = os.path.join(RESULTS_PATH, 'eval_static')
    sender_log = os.path.join(working_dir, 'eval_sender.log')
    context = StreamingContext()
    for line in open(sender_log).readlines():
        parse_line(line, context)
    analyze_stream(context, output_dir=working_dir)


if __name__ == "__main__":
    main()
