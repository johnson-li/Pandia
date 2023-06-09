import os
import re
import time
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
from pandia import RESULTS_PATH, DIAGRAMS_PATH


CODEC_NAMES = ['Generic', 'VP8', 'VP9', 'AV1', 'H264', 'Multiplex']
OUTPUT_DIR = DIAGRAMS_PATH
kDeltaTick=.25  # In ms
kBaseTimeTick=kDeltaTick*(1<<8)  # In ms
kTimeWrapPeriod=kBaseTimeTick * (1 << 24)  # In ms
PACKET_TYPES = ['audio', 'video', 'rtx', 'fec', 'padding']


def divide(a: np.ndarray, b: np.ndarray):
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


class PacketContext(object):
    def __init__(self, rtp_id) -> None:
        self.sent_at: float = -1
        self.sent_at_utc: float = -1
        self.acked_at: float = -1
        self.received_at: float = -1  # Remote ts
        self.rtp_id: int = rtp_id
        self.seq_num: int = -1
        self.payload_type = -1
        self.frame_id = -1
        self.size = -1
        self.allow_retrans: bool = None
        self.retrans_ref: int = None
        self.packet_type: str
        self.received: bool = None  # The reception is reported by RTCP transport feedback, which is biased because the feedback may be lost.

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
        self.assembled_at_utc = .0
        self.decoding_at_utc = .0
        self.decoded_at_utc = .0
        self.decoded_at = .0
        self.decoding_at = .0
        self.bitrate = 0
        self.codec = 0
        self.encoded_size = 0
        self.sequence_range = [10000000, 0]
        self.encoded_shape: tuple = (0, 0)
        self.rtp_packets: dict = {}
        self.dropped_by_encoder = False
        self.is_key_frame = False
        self.qp = 0
        self.retrans_record = {}

    def seq_len(self):
        if self.sequence_range[1] > 0:
            return self.sequence_range[1] - self.sequence_range[0] + 1
        else:
            return 0

    def last_rtp_send_ts_including_rtx(self):
        packets = list(filter(lambda p: p.packet_type in ['video', 'fec', 'rtx'], self.rtp_packets.values()))
        return max([p.sent_at for p in packets]) if packets else -1

    def last_rtp_send_ts(self):
        packets = list(filter(lambda p: p.packet_type in ['video', 'fec'], self.rtp_packets.values()))
        return max([p.sent_at for p in packets]) if packets else -1

    def packets_sent(self):
        return self.rtp_packets

    def packets_video(self):
        return list(filter(lambda p: p.packet_type in ['video'], self.rtp_packets.values()))

    def packets_rtx(self):
        return list(filter(lambda p: p.packet_type in ['rtx'], self.rtp_packets.values()))

    def packets_fec(self):
        return list(filter(lambda p: p.packet_type in ['fec'], self.rtp_packets.values()))

    def packets_padding(self):
        return list(filter(lambda p: p.packet_type in ['padding'], self.rtp_packets.values()))

    def last_rtp_recv_ts_including_rtx(self):
        packets = list(filter(lambda p: p.packet_type in ['video', 'fec', 'rtx'], self.rtp_packets.values()))
        return max([p.acked_at for p in packets]) if packets else -1

    def encoding_delay(self):
        return self.encoded_at - self.captured_at if self.encoded_at > 0 else -1

    def assemble_delay(self, utc_offset=.0):
        # return self.assembled0_at - self.captured_at if self.assembled0_at > 0 else -1
        if self.assembled_at_utc > 0:
            return self.assembled_at_utc - self.captured_at_utc - utc_offset
        else:
            return -1

    def pacing_delay(self):
        return self.last_rtp_send_ts() - self.captured_at if self.last_rtp_send_ts() else -1

    def pacing_delay_including_rtx(self):
        return self.last_rtp_send_ts_including_rtx() - self.captured_at if self.last_rtp_send_ts_including_rtx() else -1

    def decoding_queue_delay(self, utc_offset=.0):
        # return self.decoding_at - self.captured_at if self.decoding_at > 0 else -1
        if self.decoding_at_utc > 0:
            return self.decoding_at_utc - self.captured_at_utc - utc_offset
        else:
            return -1

    def decoding_delay(self, utc_offset=.0):
        # return self.decoded_at - self.captured_at if self.decoded_at > 0 else -1
        if self.decoded_at_utc > 0:
            return self.decoded_at_utc - self.captured_at_utc - utc_offset 
        else:
            return -1

    def g2g_delay(self, utc_offset=.0):
        return self.decoding_delay(utc_offset)

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
        self.start_ts: float = 0
        self.frames: Dict[int, FrameContext] = {}
        self.packets: Dict[int, PacketContext] = {}
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
        self.utc_offset: float = 0  # The difference between the sender UTC and the receiver UTC

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
        utc_ts = int(m[6]) / 1000
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
        if not context.codec_initiated:
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
        rtp_type = PACKET_TYPES[int(m[4])]  # 0: audio, 1: video, 2: rtx, 3: fec, 4: padding
        retrans_seq_num = int(m[5])
        allow_retrans = int(m[6]) != 0
        packet = PacketContext(rtp_id)
        context.packets[rtp_id] = packet
        packet.packet_type = rtp_type
        packet.frame_id = frame_id
        packet.allow_retrans = allow_retrans
        packet.retrans_ref = retrans_seq_num
        if rtp_type == 'rtx':
            packet.frame_id = context.packets[context.packet_id_map[retrans_seq_num]].frame_id
        if packet.frame_id > 0 and frame_id in context.frames:
            frame: FrameContext = context.frames[frame_id]
            frame.rtp_packets[rtp_id] = packet
            if rtp_type == 'rtx':
                original_rtp_id = context.packet_id_map[retrans_seq_num]
                frame.retrans_record.setdefault(original_rtp_id, []).append(rtp_id)
    elif 'Start encoding' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\].*Start encoding, frame id: (\\d+), shape: (\\d+) x (\\d+), bitrate: (\\d+) kbps.*'), line)
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
        assert context.packets[rtp_id].seq_num == -1
        context.packets[rtp_id].seq_num = sequence_number
        if context.packets[rtp_id].packet_type == 'video' and context.packets[rtp_id].frame_id in context.frames:
            frame = context.frames[context.packets[rtp_id].frame_id]
            frame.sequence_range[0] = min(frame.sequence_range[0], sequence_number)
            frame.sequence_range[1] = max(frame.sequence_range[1], sequence_number)
    elif 'RTCP RTT' in line:
        m = re.match(re.compile('.*\\[(\\d+)\\] RTCP RTT: (\\d+) ms.*'), line)
        ts = int(m[1]) / 1000
        rtt = int(m[2]) / 1000
        context.rtt_data.append((ts, rtt))
    # elif 'ReSendPacket' in line:
    #     m = re.match(re.compile('.*\\[(\\d+)\\] ReSendPacket, id: (\\d+).*'), line)
    #     ts = int(m[1]) / 1000
    #     sequence_number = int(m[2])
    #     if sequence_number in context.packet_id_map:
    #         rtp_id = context.packet_id_map[sequence_number]
    #         packet = context.packets[rtp_id]
    elif 'OnSentPacket' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] OnSentPacket, id: (-?\\d+), type: (\\d+), size: (\\d+), utc: (\\d+) ms.*'), line)
        ts = int(m[1]) / 1000
        rtp_id = int(m[2])
        payload_type = int(m[3])
        size = int(m[4])
        utc = int(m[5]) / 1000
        if rtp_id >= 0:
            packet: PacketContext = context.packets[rtp_id]
            packet.payload_type = payload_type
            packet.size = size
            packet.sent_at = ts 
            packet.sent_at_utc = utc
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
        if rtp_id in context.packets:
            packet = context.packets[rtp_id]
            packet.received_at = received_at
    elif 'Packet acked' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] Packet acked, id: (\\d+), received: (\\d+), delta_sum: (-?\\d+).*'), line)
        ts = int(m[1]) / 1000
        rtp_id = int(m[2])
        received = int(m[3])
        delta = int(m[4]) / 1000
        if rtp_id in context.packets:
            packet: PacketContext = context.packets[rtp_id]
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
    elif 'NTP response' in line:
        m = re.match(re.compile(
            '.*NTP response: precision: (-?[.0-9]+), offset: (-?[.0-9]+), rtt: (-?[.0-9]+).*'), line)
        precision = float(m[1])
        offset = float(m[2])
        rtt = float(m[3])
        context.utc_offset = offset
    elif 'Frame decoding acked' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] Frame decoding acked, id: (\\d+), receiving ts: (\\d+), decoding ts: (\\d+), decoded ts: (\\d+).*'), line)
        ts = int(m[1]) / 1000
        rtp_sequence = int(m[2])
        received_ts_utc = int(m[3]) / 1000
        decoding_ts_utc = int(m[4]) / 1000
        decoded_ts_utc = int(m[5]) / 1000
        rtp_id = context.packet_id_map.get(rtp_sequence, -1)
        if rtp_id > 0 and rtp_id in context.packets:
            frame_id = context.packets[rtp_id].frame_id
            if frame_id in context.frames:
                frame: FrameContext = context.frames[frame_id]
                frame.assembled_at_utc = received_ts_utc
                frame.decoding_at_utc = decoding_ts_utc
                frame.decoded_at_utc = decoded_ts_utc
                frame.decoded_at = ts
                frame.decoding_at = ts - (decoded_ts_utc - decoding_ts_utc) / 1000
                frame.assembled0_at = ts - (decoded_ts_utc - received_ts_utc) / 1000
                context.last_decoded_frame_id = \
                        max(frame.frame_id, context.last_decoded_frame_id)
    elif 'Frame reception acked' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] Frame reception acked, id: (\\d+).*'), line)
        ts = int(m[1]) / 1000
        rtp_sequence = int(m[2])
        rtp_id = context.packet_id_map[rtp_sequence]
        if rtp_id in context.packets:
            frame_id = context.packets[rtp_id].frame_id
            if frame_id in context.frames:
                context.frames[frame_id].assembled_at = ts
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
                                     'assembled by': frame.assemble_delay(context.utc_offset),
                                     'queued by': frame.decoding_queue_delay(context.utc_offset),
                                     'decoded by': frame.decoding_delay(context.utc_offset),})
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
    # plt.ylim([0, ylim])
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
    ax1.plot(lost_frames, [10 for _ in lost_frames], 'xb')
    ax1.plot([f[0] for f in key_frames], [f[1] for f in key_frames], 'o')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xlabel('Timestamp (s)')
    ax1.set_ylabel('Encoded size (KB)')
    ax1.legend(['Encoded size', 'Lost frames', 'Key frames'])
    ax2 = ax1.twinx()
    ax2.plot(x, bitrates, 'r')
    ax2.set_ylabel('Bitrate (Kbps)')
    ax2.tick_params(axis='y', labelcolor='r')
    plt.savefig(os.path.join(output_dir, 'mea-size-frame.pdf'))

    plt.close()
    x = []
    y = []
    yy = []
    for frame in context.frames.values():
        count_all = len(frame.packets_sent())
        count_lost = len(frame.packets_rtx())
        x.append(frame.captured_at - context.start_ts)
        y.append(count_lost / count_all * 100 if count_all > 0 else 0)
        yy.append(count_lost)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, y, 'b')
    ax1.tick_params(axis='y', labelcolor='b')
    x = np.array(x)
    yy = np.array(yy)
    # ax2.plot(x[yy > 0], yy[yy > 0], 'r.')
    ax2.plot(x, yy, 'r.')
    plt.xlabel('Timestamp (s)')
    ax1.set_ylabel('Packet loss rate per frame (%)')
    ax2.set_ylabel('Number of lost packets per frame')
    ax2.tick_params(axis='y', labelcolor='r')
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
    ax1.plot(x, accu, 'b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xlabel('Timestamp (s)')
    ax1.set_ylabel('Accumulated frame size (KB)')
    ax2 = ax1.twinx()
    ax2.plot(x, fps, 'r')
    ax2.set_ylabel('FPS')
    ax2.tick_params(axis='y', labelcolor='r')
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
    for pkt in sorted(context.packets.values(), key=lambda x: x.sent_at):
        pkt: PacketContext = pkt
        if pkt.ack_delay() > 0:
            data_ack.append((pkt.sent_at, pkt.ack_delay()))
        if pkt.recv_delay() != -1:
            data_recv.append((pkt.sent_at, pkt.recv_delay() - context.utc_offset))
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
    y = divide(bucks_lost, bucks_sent) * 100
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
            if p.packet_type == 'rtx':
                bucks_retrans[i] += 1
    x = [i * duration for i in range(len(bucks_sent))]
    y = divide(bucks_retrans, bucks_sent) * 100
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

    plt.close()
    data = [[[], []], [[], []], [[], []]]  # [rtp_video, rtp_rtx, rtp_fec]
    for f in context.frames.values():
        for pkt in f.rtp_packets.values():
            d = None
            if pkt.packet_type == 'video':
                d = data[0]
            elif pkt.packet_type == 'rtx':
                d = data[1]
            elif pkt.packet_type == 'fec':
                d = data[2]
            if d and pkt.sent_at > 0:
                d[0].append(f.captured_at - context.start_ts)
                d[1].append((pkt.sent_at - f.encoded_at) * 1000)
    for d, m in zip(data, ['.', '8', '^']):
        plt.plot(d[0], d[1], m)
    plt.xlabel('Frame timestamp (s)')
    plt.ylabel('RTP egress timestamp (ms)')
    plt.ylim([0, 150])
    plt.legend(['Video', 'RTX', 'FEC'])
    plt.savefig(os.path.join(output_dir, 'mea-rtp-pacing-ts.pdf'))


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
    period = .001
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


def get_stream_context(result_path=os.path.join(RESULTS_PATH, 'eval_static')) -> StreamingContext:
    sender_log = os.path.join(result_path, 'eval_sender.log')
    context = StreamingContext()
    for line in open(sender_log).readlines():
        try:
            parse_line(line, context)
        except Exception as e:
            print(f"Error parsing line: {line}")
            raise e
    return context


def main(result_path=os.path.join(RESULTS_PATH, 'eval_static')) -> None:
    sender_log = os.path.join(result_path, 'eval_sender.log')
    context = StreamingContext()
    for line in open(sender_log).readlines():
        try:
            parse_line(line, context)
        except Exception as e:
            print(f"Error parsing line: {line}")
            raise e
    analyze_stream(context, output_dir=result_path)


if __name__ == "__main__":
    main()
