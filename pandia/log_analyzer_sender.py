import argparse
import os
import re
import time
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from pandia import RESULTS_PATH, DIAGRAMS_PATH
from pandia.agent.normalization import RESOLUTION_LIST


CODEC_NAMES = ['Generic', 'VP8', 'VP9', 'AV1', 'H264', 'Multiplex']
OUTPUT_DIR = DIAGRAMS_PATH
kDeltaTick=.25  # In ms
kBaseTimeTick=kDeltaTick*(1<<8)  # In ms
kTimeWrapPeriod=kBaseTimeTick * (1 << 24)  # In ms
PACKET_TYPES = ['audio', 'video', 'rtx', 'fec', 'padding']
FIG_EXTENSION = 'png'
DPI = 600


def divide(a: np.ndarray, b: np.ndarray):
    return np.divide(a.astype(np.float32), b.astype(np.float32), 
                     out=np.zeros_like(a, dtype=np.float32), where=b != 0)


class PacketContext(object):
    def __init__(self, rtp_id) -> None:
        self.sent_at: float = -1
        self.sent_at_utc: float = -1
        self.acked_at: float = -1
        self.received_at_utc: float = -1  # Remote ts
        self.rtp_id: int = rtp_id
        self.seq_num: int = -1
        self.payload_type = -1
        self.frame_id = -1
        self.size = -1
        self.first_packet_in_frame = False  # The first RTP video packet for a frame
        self.last_packet_in_frame = False  # The last RTP video packet for a frame
        self.allow_retrans: Optional[bool] = None
        self.retrans_ref: Optional[int] = None
        self.packet_type: str
        self.received: Optional[bool] = None  # The reception is reported by RTCP transport feedback, which is biased because the feedback may be lost.

    def ack_delay(self):
        return self.acked_at - self.sent_at if self.acked_at > 0 else -1

    def recv_delay(self, utc_offset=.0):
        return self.received_at_utc - self.sent_at_utc - utc_offset if self.received_at_utc > 0 else -1


class FecContext(object):
    def __init__(self) -> None:
        self.fec_key_data = []
        self.fec_delta_data = []


class FrameContext(object):
    def __init__(self, frame_id, captured_at) -> None:
        self.frame_id = frame_id
        self.captured_at = captured_at
        self.captured_at_utc = .0
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
        self.fps = 0
        self.encoded_size = 0
        self.sequence_range = [10000000, 0]
        self.encoded_shape: tuple = (0, 0)
        self.rtp_packets: dict = {}
        self.dropped_by_encoder = False
        self.is_key_frame = False
        self.qp = 0
        self.retrans_record = {}

    @property
    def encoded(self):
        return self.encoded_at > 0

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

    def packets_video(self) -> List[PacketContext]:
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
    def __init__(self, monitor_durations=[1]) -> None:
        self.start_ts: float = 0
        self.frames: Dict[int, FrameContext] = {}
        self.packets: Dict[int, PacketContext] = {}
        self.packet_id_map = {}
        self.networking = NetworkContext()
        self.fec = FecContext()
        # self.bitrate_data = []
        self.rtt_data = []
        self.packet_loss_data = []
        self.fps_data = []
        self.pacing_queue_data = []
        self.codec: Optional[int] = None
        self.last_captured_frame_id = 0
        self.last_decoded_frame_id = 0
        self.last_egress_packet_id = 0
        self.last_acked_packet_id = 0
        self.action_context = ActionContext()
        self.utc_local_offset: float = 0 # The difference between the sender local time and thesender UTC time 
        self.utc_offset: float = 0  # The difference between the sender UTC and the receiver UTC
        self.monitor_blocks = {d: MonitorBlock(d) for d in monitor_durations}
        self.last_ts = .0
        self.drl_bitrate = []
        self.drl_pacing_rate = []

    def update_utc_offset(self, offset):
        self.utc_offset = offset
        print(f'Update utc offset to {offset}')
        for mb in self.monitor_blocks.values():
            mb.update_utc_offset(offset)

    def update_utc_local_offset(self, offset):
        self.utc_local_offset = offset
        for mb in self.monitor_blocks.values():
            mb.update_utc_local_offset(offset)

    def update_ts(self, ts: float):
        self.last_ts = ts
        for mb in self.monitor_blocks.values():
            mb.ts = ts

    def reset_step_context(self):
        self.action_context = ActionContext()

    @property
    def codec_initiated(self):
        return self.start_ts > 0


class MonitorBlockData(object):
    def __init__(self, ts_fn, val_fn, duration=1, val_checker=lambda v: v >= 0, ts_offset=0) -> None:
        self.duration = duration
        self.data = []
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
        # Prevent the data from being empty
        # It is useful when the measured latency is larger than the duration
        while self.num > 1 and self.ts(self.data[0]) < ts - self.duration:
            val = self.data.pop(0)
            self.num -= 1
            self.sum -= self.val_fn(val)

    def avg(self):
        return self.sum / self.num if self.num > 0 else 0


class MonitorBlock(object):
    def __init__(self, duration=1) -> None:
        self.ts: float = 0
        self.duration: float = duration
        self.utc_offset = 0
        # Frame statictics
        self.frame_encoded_size = MonitorBlockData(lambda f: f.encoded_at, lambda f: f.encoded_size, duration=duration)
        self.frame_acked_size = MonitorBlockData(lambda f: f.acked_at, lambda f: f.encoded_size, duration=duration)
        self.frame_qp_data = MonitorBlockData(lambda f: f.encoded_at, lambda f: f.qp, duration=duration)
        self.frame_height_data = MonitorBlockData(lambda f: f.encoding_at, lambda f: f.height, duration=duration)
        self.frame_encoded_height_data = MonitorBlockData(lambda f: f.encoded_at, lambda f: f.encoded_shape[1], duration=duration)
        self.frame_encoding_delay_data = MonitorBlockData(lambda f: f.encoded_at, lambda f: f.encoding_delay(), duration=duration)
        self.frame_egress_delay_data = MonitorBlockData(lambda f: f.encoded_at, lambda f: f.pacing_delay(), duration=duration)
        self.frame_recv_delay_data = MonitorBlockData(lambda f: f.encoded_at, lambda f: f.assemble_delay(self.utc_offset), duration=duration)
        self.frame_decoding_delay_data = MonitorBlockData(lambda f: f.encoded_at, lambda f: f.decoding_queue_delay(self.utc_offset), duration=duration)
        self.frame_decoded_delay_data = MonitorBlockData(lambda f: f.encoded_at, lambda f: f.decoding_delay(self.utc_offset), duration=duration)
        self.frame_key_counter = MonitorBlockData(lambda f: f.encoded_at, lambda f: 1 if f.is_key_frame else 0, duration=duration)
        # Packet statistics
        self.pkts_sent_size = MonitorBlockData(lambda p: p.sent_at, lambda p: p.size, duration=duration)
        self.pkts_acked_size = MonitorBlockData(lambda p: p.acked_at, lambda p: p.size, duration=duration)
        self.pkts_trans_delay_data = MonitorBlockData(lambda p: p.sent_at, lambda p: p.recv_delay(self.utc_offset), duration=duration)
        self.pkts_lost_count = MonitorBlockData(lambda p: p.acked_at, lambda p: 1 if not p.received else 0, duration=duration)
        self.pkts_delay_interval_data = MonitorBlockData(lambda s: s[0], lambda s: s[1], duration=duration)
        # Setting statistics
        self.pacing_rate_data = MonitorBlockData(lambda p: p[0], lambda p: p[1], duration=duration)
        self.bitrate_data = MonitorBlockData(lambda f: f.encoded_at, lambda f: f.bitrate, duration=duration)

    def update_utc_local_offset(self, offset):
        self.pkts_acked_size.ts_offset = offset

    def update_utc_offset(self, offset):
        self.utc_offset = offset

    @property
    def action_gap(self):
        return self.bitrate - self.pacing_rate

    @property
    def frame_key_count(self):
        return self.frame_key_counter.sum

    @property
    def pacing_rate(self):
        return self.pacing_rate_data.avg()

    @property
    def frame_fps(self):
        return self.frame_encoded_size.num / self.duration

    # Frame bitrate is calculated based on the encoded frame size
    @property
    def frame_bitrate(self):
        return self.frame_encoded_size.sum * 8 / self.duration

    # Bitrate is the value set to the encoder before encoding
    @property
    def bitrate(self):
        return self.bitrate_data.avg()

    @property
    def frame_encoding_delay(self):
        return self.frame_encoding_delay_data.avg()

    @property
    def frame_decoded_delay(self):
        return self.frame_decoded_delay_data.avg()

    @property
    def frame_decoding_delay(self):
        return self.frame_decoding_delay_data.avg()

    @property
    def frame_encoded_height(self):
        return self.frame_encoded_height_data.avg()

    @property
    def frame_height(self):
        return self.frame_height_data.avg()

    @property
    def frame_qp(self):
        return self.frame_qp_data.avg()

    @property
    def frame_recv_delay(self):
        return self.frame_recv_delay_data.avg()

    @property
    def frame_egress_delay(self):
        return self.frame_egress_delay_data.avg()

    @property
    def frame_size(self):
        return self.frame_encoded_size.avg()

    @property
    def pkt_egress_rate(self):
        return self.pkts_sent_size.sum * 8 / self.duration

    @property
    def pkt_trans_delay(self):
        return self.pkts_trans_delay_data.avg()

    @property
    def pkt_ack_rate(self):
        return self.pkts_acked_size.sum * 8 / self.duration

    @property
    def pkt_loss_rate(self):
        return self.pkts_lost_count.sum / self.pkts_sent_size.num if self.pkts_sent_size.num > 0 else 0

    @property
    def pkt_delay_interval(self):
        return self.pkts_delay_interval_data.avg()

    def on_packet_added(self, packet: PacketContext, ts: float):
        pass

    def on_packet_sent(self, packet: PacketContext, frame: Optional[FrameContext], ts: float):
        self.pkts_sent_size.append(packet, ts)
        if packet.last_packet_in_frame and frame:
            self.frame_egress_delay_data.append(frame, ts)

    def on_packet_acked(self, packet: PacketContext, packet_pre: Optional[PacketContext], ts: float):
        if packet.received:
            # print(f'Packet {packet.rtp_id} acked of {packet.size} bytes, ts: {ts}')
            self.pkts_acked_size.append(packet, ts)
            # if packet.recv_delay() > 1:
            #     print(packet.recv_delay(), packet.sent_at_utc, packet.received_at_utc, packet.received_at_utc - packet.sent_at_utc, packet.rtp_id, packet.seq_num)
            self.pkts_trans_delay_data.append(packet, ts)
        if packet.received is not None:
            self.pkts_lost_count.append(packet, ts)
        if packet_pre:
            self.pkts_delay_interval_data\
                .append((ts, (packet.received_at_utc - packet_pre.received_at_utc) - 
                         (packet.sent_at - packet_pre.sent_at)), ts)

    def on_frame_added(self, frame: FrameContext, ts: float):
        pass

    def on_frame_encoding(self, frame: FrameContext, ts: float):
        self.frame_height_data.append(frame, ts)
        self.bitrate_data.append(frame, ts)

    def on_frame_encoded(self, frame: FrameContext, ts: float):
        self.frame_encoded_size.append(frame, ts)
        self.frame_qp_data.append(frame, ts)
        self.frame_encoded_height_data.append(frame, ts)
        self.frame_encoding_delay_data.append(frame, ts)
        self.frame_key_counter.append(frame, ts)

    def on_frame_decoding_updated(self, frame: FrameContext, ts: float):
        self.frame_recv_delay_data.append(frame, ts)
        self.frame_decoding_delay_data.append(frame, ts)
        self.frame_decoded_delay_data.append(frame, ts)

    def on_pacing_rate_set(self, ts: float, rate: float):
        self.pacing_rate_data.append((ts, rate), ts)


class NetworkContext(object):
    def __init__(self) -> None:
        self.pacing_rate_data = []
        self.pacing_burst_interval_data = []


def parse_line(line, context: StreamingContext) -> dict:
    data = {}
    ts = 0
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
        if context.utc_local_offset == 0:
            context.update_utc_local_offset(utc_ts - ts)
        [mb.on_frame_added(frame, ts) for mb in context.monitor_blocks.values()]
    elif 'UpdateFecRates' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] UpdateFecRates, fraction lost: ([0-9.]+).*'), line)
        ts = int(m[1]) / 1000
        loss_rate = float(m[2])
        context.packet_loss_data.append((ts, loss_rate))
    elif 'SetupCodec' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] SetupCodec, config, codec type: (\\d+).*'), line)
        ts = int(m[1]) / 1000
        codec = int(m[2])
        if context.codec is None:
            context.codec = codec
            context.start_ts = ts
    elif 'Egress paused because of pacing rate constraint' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] Egress paused because of pacing rate constraint, left packets: (\\d+).*'), line)
        ts = int(m[1]) / 1000
        left_packets = int(m[2])
        context.pacing_queue_data.append((ts, left_packets))
    elif ' SendPacket,' in line:
        m = re.match('.*\\[(\\d+)\\].*SendPacket, id: (\\d+), seq: (\\d+)'
                     ', first in frame: (\\d+), last in frame: (\\d+), fid: (\\d+), type: (\\d+)'
                     ', rtx seq: (\\d+), allow rtx: (\\d+), size: (\\d+).*', line)
        ts = int(m[1]) / 1000
        rtp_id = int(m[2])
        seq_num = int(m[3])
        first_in_frame = int(m[4]) == 1
        last_in_frame = int(m[5]) == 1
        frame_id = int(m[6])
        rtp_type = PACKET_TYPES[int(m[7])]  # 0: audio, 1: video, 2: rtx, 3: fec, 4: padding
        retrans_seq_num = int(m[8])
        allow_retrans = int(m[9]) != 0
        size = int(m[10])
        if rtp_id > 0:
            packet = PacketContext(rtp_id)
            packet.seq_num = seq_num
            packet.packet_type = rtp_type
            packet.frame_id = frame_id
            packet.first_packet_in_frame = first_in_frame
            packet.last_packet_in_frame = last_in_frame
            packet.allow_retrans = allow_retrans
            packet.retrans_ref = retrans_seq_num
            packet.size = size
            context.packets[rtp_id] = packet
            context.packet_id_map[seq_num] = rtp_id
            [mb.on_packet_added(packet, ts) for mb in context.monitor_blocks.values()]
            if rtp_type == 'rtx':
                packet.frame_id = context.packets[context.packet_id_map[retrans_seq_num]].frame_id
            if packet.frame_id > 0 and frame_id in context.frames:
                frame: FrameContext = context.frames[frame_id]
                frame.rtp_packets[rtp_id] = packet
                if rtp_type == 'rtx':
                    original_rtp_id = context.packet_id_map[retrans_seq_num]
                    frame.retrans_record.setdefault(original_rtp_id, []).append(rtp_id)
                if rtp_type == 'video':
                    frame.sequence_range[0] = min(frame.sequence_range[0], seq_num)
                    frame.sequence_range[1] = max(frame.sequence_range[1], seq_num)
    elif 'Start encoding' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\].*Start encoding, frame id: (\\d+), shape: (\\d+) x (\\d+)'
            ', bitrate: (\\d+) kbps, key frame: (\\d+), fps: (\\d+).*'), line)
        ts = int(m[1]) / 1000
        frame_id = int(m[2])
        width = int(m[3])
        height = int(m[4])
        bitrate = int(m[5]) * 1024
        key_frame = int(m[6]) == 1
        fps = int(m[7])
        context.action_context.resolution = width
        if context.action_context.bitrate <= 0:
            context.action_context.bitrate = bitrate
        if frame_id in context.frames:
            frame: FrameContext = context.frames[frame_id]
            frame.bitrate = bitrate
            frame.fps = fps
            frame.width = width
            frame.height = height
            frame.encoding_at = ts
            # context.bitrate_data.append([ts, bitrate])
            # context.fps_data.append([ts, fps])
            [mb.on_frame_encoding(frame, ts) for mb in context.monitor_blocks.values()]
    elif 'Finish encoding' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] Finish encoding, frame id: (\\d+), frame type: (\\d+), frame shape: (\\d+)x(\\d+), frame size: (\\d+), is key: (\\d+), qp: (-?\\d+).*'), line)
        ts = int(m[1]) / 1000
        frame_id = int(m[2])
        frame_type = int(m[3])
        width= int(m[4])
        height = int(m[5])
        frame_size = int(m[6])
        is_key = int(m[7])
        qp = int(m[8])
        frame = context.frames[frame_id]
        frame.encoded_at = ts
        frame.encoded_shape = (width, height)
        frame.encoded_size = frame_size
        frame.qp = qp
        frame.is_key_frame = is_key != 0
        [mb.on_frame_encoded(frame, ts) for mb in context.monitor_blocks.values()]
    elif 'RTCP RTT' in line:
        m = re.match(re.compile('.*\\[(\\d+)\\] RTCP RTT: (\\d+) ms.*'), line)
        ts = int(m[1]) / 1000
        rtt = int(m[2]) / 1000
        context.rtt_data.append((ts, rtt))
    elif 'SendVideo' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] SendVideo, frame id: (\\d+), number of RTP packets: (\\d+).*'), line)
        ts = int(m[1]) / 1000
        frame_id = int(m[2])
        num_rtp_packets = int(m[3])
        context.frames[frame_id].num_rtp_packets = num_rtp_packets
    elif 'OnSentPacket' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] OnSentPacket, id: (-?\\d+), type: (\\d+), size: (\\d+), utc: (\\d+) ms.*'), line)
        ts = int(m[1]) / 1000
        rtp_id = int(m[2])
        payload_type = int(m[3])
        size = int(m[4])
        utc = int(m[5]) / 1000
        if rtp_id > 0:
            packet: PacketContext = context.packets[rtp_id]
            packet.payload_type = payload_type
            packet.size = size
            packet.sent_at = ts 
            packet.sent_at_utc = utc
            context.last_egress_packet_id = max(rtp_id, context.last_egress_packet_id)
            [mb.on_packet_sent(packet, context.frames.get(packet.frame_id, None), ts) for mb in context.monitor_blocks.values()]
    elif 'RTCP feedback,' in line:
        m = re.match(re.compile('.*\\[(\\d+)\\] RTCP feedback.*'), line)
        ts = int(m[1]) / 1000
        ms = re.findall(r'packet (acked|lost): (\d+) at (\d+) ms', line)
        pkt_pre = None
        for ack_type, rtp_id, received_at in ms:
            # The recv time is wrapped by kTimeWrapPeriod.
            # The fixed value 1570 should be calculated according to the current time.
            offset = int(time.time() * 1000 / kTimeWrapPeriod) - 1
            received_at = (int(received_at) + offset * kTimeWrapPeriod) / 1000
            rtp_id = int(rtp_id)
            # received_at = int(received_at) / 1000
            if rtp_id in context.packets:
                packet = context.packets[rtp_id]
                packet.received_at_utc = received_at
                packet.received = ack_type == 'acked'
                packet.acked_at = ts
                context.last_acked_packet_id = \
                    max(rtp_id, context.last_acked_packet_id)
                [mb.on_packet_acked(packet, pkt_pre, ts) for mb in context.monitor_blocks.values()]
                if packet.received:
                    pkt_pre = packet
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
                [mb.on_frame_decoding_updated(frame, ts) for mb in context.monitor_blocks.values()]
    elif 'Apply bitrate' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] Apply bitrate: (\\d+) kbps, pacing rate: (\\d+) kbps from shared memory.*'), line)
        ts = int(m[1]) / 1000
        bitrate = int(m[2]) * 1024
        pacing_rate = int(m[3]) * 1024
        context.drl_bitrate.append((ts, bitrate))
        context.drl_pacing_rate.append((ts, pacing_rate))
    elif 'SetPacingRates' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] SetPacingRates, pacing rate: (\\d+) kbps, pading rate: (\\d+) kbps.*'), line)
        ts = int(m[1]) / 1000
        pacing_rate = int(m[2])
        padding_rate = int(m[3])
        context.action_context.pacing_rate = pacing_rate
        context.networking.pacing_rate_data.append(
            [ts, pacing_rate, padding_rate])
        [mb.on_pacing_rate_set(ts, pacing_rate * 1024) for mb in context.monitor_blocks.values()]
    # elif 'SetRates, ' in line:
    #     m = re.match(re.compile(
    #         '.*\\[(\\d+)\\] SetRates, stream id: (\\d+), bitrate: (\\d+) kbps, '
    #         'max bitrate: (\\d+) kbps, framerate: (\\d+).*'), line)
    #     ts = int(m[1]) / 1000
    #     stream_id = int(m[2]) 
    #     bitrate = int(m[3])
    #     bitrate_max = int(m[4])
    #     fps = int(m[5])
    #     context.bitrate_data.append([ts, bitrate, bitrate_max])
    #     context.fps_data.append([ts, fps])
    if ts:
        context.update_ts(ts)
    return data


def analyze_frame(context: StreamingContext, output_dir=OUTPUT_DIR) -> None:
    frame_id_list = list(sorted(context.frames.keys()))
    frames = [context.frames[i] for i in frame_id_list]
    frames_encoded: List[FrameContext] = list(filter(lambda f: f.encoded_size > 0, frames))
    frames_encoded_ts = np.array([f.captured_at - context.start_ts for f in frames_encoded])
    frames_dropped = list(filter(lambda f: f.encoded_size <= 0, frames))
    frames_key = list(filter(lambda f: f.is_key_frame, frames))
    if (len(frames_encoded) == 0):
        print('ERROR: No frame encoded.')
        return

    plt.close()
    plt.plot([f.captured_at - context.start_ts for f in frames], [f.frame_id for f in frames], '.')
    plt.xlabel('Frame capture time (s)')
    plt.ylabel('Frame ID')
    plt.savefig(os.path.join(output_dir, f'mea-frame-id.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    decoding_delay = np.array([f.decoding_delay(context.utc_offset) * 1000 for f in frames_encoded])
    decoding_queue_delay = np.array([f.decoding_queue_delay(context.utc_offset) * 1000 for f in frames_encoded])
    assemble_delay = np.array([f.assemble_delay(context.utc_offset) * 1000 for f in frames_encoded])
    pacing_delay = np.array([f.pacing_delay() * 1000 for f in frames_encoded])
    encoding_delay = np.array([f.encoding_delay() * 1000 for f in frames_encoded])
    plt.plot(frames_encoded_ts[decoding_delay >= 0], decoding_delay[decoding_delay >= 0], 'b')
    plt.plot(frames_encoded_ts[decoding_queue_delay >= 0], decoding_queue_delay[decoding_queue_delay >= 0], 'g')
    plt.plot(frames_encoded_ts[assemble_delay >= 0], assemble_delay[assemble_delay >= 0], 'r')
    plt.plot(frames_encoded_ts[pacing_delay >= 0], pacing_delay[pacing_delay >= 0], 'c')
    plt.plot(frames_encoded_ts[pacing_delay >= 0], pacing_delay[pacing_delay >= 0], 'y')
    plt.plot(frames_encoded_ts[encoding_delay >= 0], encoding_delay[encoding_delay >= 0], 'm')
    plt.legend(['Decoding', 'Decoding queue', 'Transmission', 'Pacing (RTX)', 'Pacing', 'Encoding'])
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Delay (ms)')
    # plt.xlim([0, 10])
    # plt.ylim([0, 50])
    plt.savefig(os.path.join(output_dir, f'mea-delay-frame.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    res_list = [f.encoded_shape[1] for f in frames_encoded]
    res_list = list(sorted(set(res_list)))
    plt.plot(frames_encoded_ts, [res_list.index(f.encoded_shape[1]) for f in frames_encoded])
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Resolution')
    plt.yticks(range(len(res_list)), [str(r) for r in res_list])
    plt.savefig(os.path.join(output_dir, f'mea-resolution-frame.{FIG_EXTENSION}'), dpi=DPI)
    
    plt.close()
    fig, ax1 = plt.subplots()
    ax1.plot(frames_encoded_ts, [f.encoded_size / 1024 for f in frames_encoded], 'b.')
    ax1.plot([f.captured_at - context.start_ts for f in frames_dropped], [10 for _ in frames_dropped], 'xb')
    ax1.plot([f.captured_at - context.start_ts for f in frames_key], [f.encoded_size / 1024 for f in frames_key], '^g')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xlabel('Timestamp (s)')
    ax1.set_ylabel('Encoded size (KB)')
    ax1.legend(['Encoded size', 'Dropped frames', 'Key frames'])
    ax2 = ax1.twinx()
    ax2.plot([f.captured_at - context.start_ts for f in frames_encoded], [f.bitrate for f in frames_encoded], 'r.')
    ax2.set_ylabel('Bitrate (Kbps)')
    ax2.tick_params(axis='y', labelcolor='r')
    # plt.xlim([0, frames_encoded[-1].captured_at - context.start_ts])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'mea-size-frame.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    frame_rtp_num_list = np.array([len(f.packets_video()) for f in frames_encoded])
    frame_rtp_lost_num_list = np.array([len([p for p in f.packets_video() if p.received is False]) for f in frames_encoded])
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(frames_encoded_ts, divide(frame_rtp_lost_num_list, frame_rtp_num_list) * 100, 'b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.plot(frames_encoded_ts, frame_rtp_lost_num_list, 'r.')
    plt.xlabel('Timestamp (s)')
    ax1.set_ylabel('Packet loss rate per frame (%)')
    ax2.set_ylabel('Number of lost packets per frame')
    ax2.tick_params(axis='y', labelcolor='r')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'mea-loss-packet-frame.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    duration = .5
    bucks = int((frames_encoded[-1].encoding_at - context.start_ts) / duration + 1)
    data = np.zeros((bucks, 1))
    for frame in frames_encoded:
        if frame.encoded_size > 0:
            i = int((frame.encoding_at - context.start_ts) / duration)
            data[i] += frame.encoded_size
    plt.plot(np.arange(bucks) * duration, data * 8 / duration / 1024, '.b')
    plt.ylabel('Rates (Kbps)')
    plt.plot([d[0] - context.start_ts for d in context.networking.pacing_rate_data], 
             [d[1] for d in context.networking.pacing_rate_data], '.r')
    plt.xlabel('Timestamp (s)')
    plt.legend(['Frame encoded bitrate', 'Pacing rate'])
    plt.savefig(os.path.join(output_dir, f'mea-bitrate.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Timestamp (s)')
    ax1.set_ylabel('Bitrate (Kbps)', color='b')
    drl_bitrate = np.array(context.drl_bitrate)
    # ax1.plot((drl_bitrate[:, 0] - context.start_ts), drl_bitrate[:, 1] / 1024, 'b--')
    ax1.plot([f.encoding_at - context.start_ts for f in frames_encoded], [f.bitrate / 1024 for f in frames_encoded], 'b')
    ax1.plot(np.arange(bucks) * duration, data * 8 / duration / 1024, '.b')
    ax1.legend(['Encoding bitrate', 'Encoded bitrate'])
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Encoding FPS', color='r')
    ax2.plot([f.encoding_at - context.start_ts for f in frames_encoded], [f.fps for f in frames_encoded], 'r')
    ax2.tick_params(axis='y', labelcolor='r')
    plt.savefig(os.path.join(output_dir, f'set-codec-params.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    plt.plot([f.encoded_at - context.start_ts for f in frames_encoded], [f.qp for f in frames_encoded], '.')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('QP')
    plt.savefig(os.path.join(output_dir, f'rep-qp-frame.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    duration = .2
    bucks = int((frames_encoded[-1].captured_at - context.start_ts) / duration + 1)
    data = np.zeros(bucks)
    for f in frames_encoded:
        data[int((f.captured_at - context.start_ts) / duration)] += 1
    plt.plot(np.arange(bucks) * duration, data / duration, '.')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('FPS')
    plt.savefig(os.path.join(output_dir, f'mea-fps.{FIG_EXTENSION}'), dpi=DPI)


def analyze_packet(context: StreamingContext, output_dir=OUTPUT_DIR) -> None:
    data_ack = []
    data_recv = []
    for pkt in sorted(context.packets.values(), key=lambda x: x.sent_at):
        pkt: PacketContext = pkt
        if pkt.ack_delay() > 0:
            data_ack.append((pkt.sent_at, pkt.ack_delay()))
        if pkt.recv_delay() != -1:
            data_recv.append((pkt.sent_at, pkt.recv_delay() - context.utc_offset))
    if len(data_ack) == 0:
        print('ERROR: No packet ACKed.')
        return
    plt.close()
    x = [(d[0] - context.start_ts) for d in data_recv]
    y = [d[1] * 1000 for d in data_recv]
    plt.plot(x, y, '.')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Packet transmission delay (ms)')
    if y:
        plt.ylim([min(y), max(y)])
    # plt.ylim([0, 10])
    plt.savefig(os.path.join(output_dir, f'mea-delay-packet-biased.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    plt.plot([(d[0] - context.start_ts)
             for d in data_ack], [d[1] * 1000 for d in data_ack], 'x')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('RTT (ms)')
    # plt.ylim([0, 50])
    plt.savefig(os.path.join(output_dir, f'mea-delay-packet.{FIG_EXTENSION}'), dpi=DPI)

    cdf_x = list(sorted([d[1] * 1000 for d in data_ack]))
    cdf_y = np.arange(len(cdf_x)) / len(cdf_x)
    plt.close()
    plt.plot(cdf_x, cdf_y)
    plt.xlabel('Packet ACK delay (ms)')
    plt.ylabel('CDF')
    plt.ylim([0, 1])
    plt.xlim([0, max(cdf_x)])
    plt.savefig(os.path.join(output_dir, f'mea-delay-packet-cdf.{FIG_EXTENSION}'), dpi=DPI)

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
    plt.savefig(os.path.join(output_dir, f'mea-loss-packet.{FIG_EXTENSION}'), dpi=DPI)

    for duration in [1, .1, .01, .001]:
        plt.close()
        bucks = int((packets[-1].sent_at - context.start_ts) / duration + 1)
        data = np.zeros(bucks)
        for p in packets:
            if p.sent_at >= context.start_ts:
                data[int((p.sent_at - context.start_ts) / duration)] += p.size
        plt.plot(np.arange(bucks) * duration, data / duration * 8 / 1024 / 1024, '.')
        plt.xlabel('Timestamp (s)')
        plt.ylabel('Egress rate (Mbps)')
        plt.savefig(os.path.join(output_dir, f'mea-egress-rate-{int(duration * 1000)}.{FIG_EXTENSION}'), dpi=DPI)

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
    plt.savefig(os.path.join(output_dir, f'mea-retrans-packet.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    x = [i[0] - context.start_ts for i in context.packet_loss_data]
    y = [i[1] * 100 for i in context.packet_loss_data]
    plt.plot(x, y, '.')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Packet loss rate (%)')
    plt.savefig(os.path.join(output_dir, f'rep-loss-packet.{FIG_EXTENSION}'), dpi=DPI)

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
    # plt.ylim([0, 150])
    plt.legend(['Video', 'RTX', 'FEC'])
    plt.savefig(os.path.join(output_dir, f'mea-rtp-pacing-ts.{FIG_EXTENSION}'), dpi=DPI)


def analyze_network(context: StreamingContext, output_dir=OUTPUT_DIR) -> None:
    data = context.networking.pacing_rate_data
    x = [d[0] - context.start_ts for d in data]
    y = [d[1] / 1024 for d in data]
    yy = [d[2] / 1024 for d in data]
    if len(data) == 0:
        print('ERROR: No network data.')
        return
    plt.close()
    plt.plot(x, y, '.')
    plt.plot(x, yy, '.')
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Rate (Mbps)")
    plt.legend(["Pacing rate", "Padding rate"])
    plt.savefig(os.path.join(output_dir, f'set-pacing-rate.{FIG_EXTENSION}'), dpi=DPI)
    ts_min = context.start_ts
    ts_max = max([p.sent_at for p in context.packets.values()])
    ts_range = ts_max - ts_min
    period = .1
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
    plt.savefig(os.path.join(output_dir, f'mea-sending-rate.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    x = [r[0] - context.start_ts for r in context.rtt_data]
    y = [r[1] * 1000 for r in context.rtt_data]
    plt.plot(x, y, '.')
    plt.xlabel("Timestamp (s)")
    plt.ylabel("RTT (ms)")
    plt.savefig(os.path.join(output_dir, f'rep-rtt.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    x = [d[0] - context.start_ts for d in context.pacing_queue_data]
    y = [d[1] for d in context.pacing_queue_data]
    plt.plot(x, y)
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Pacing queue size (packets)")
    plt.savefig(os.path.join(output_dir, f'mea-pacing-queue.{FIG_EXTENSION}'), dpi=DPI)


def print_statistics(context: StreamingContext) -> None:
    print("========== STATISTICS [SENDER] ==========")
    frame_ids = list(filter(lambda k: context.frames[k].encoded, context.frames.keys()))
    id_min = min(frame_ids) if frame_ids else 0
    id_max = max(frame_ids) if frame_ids else 0
    frames_total = (id_max - id_min + 1) if frame_ids else 0
    frames_sent = len(frame_ids)
    frame_ids = list(filter(lambda f: id_min <= f.frame_id <= id_max and f.decoded_at > 0, context.frames.values()))
    frames_decoded = len(frame_ids)
    loss_rate_encoding = ((frames_total - frames_sent) / frames_total) if frames_total else 0
    loss_rate_decoding = ((frames_total - frames_decoded) / frames_total) if frames_total else 0
    print(f"Total frames: {frames_total}, encoding drop rate: {loss_rate_encoding:.2%}, "
          f"decoding loss rate: {loss_rate_decoding:.2%}")


def analyze_fec(context: StreamingContext, output_dir=OUTPUT_DIR) -> None:
    plt.close()
    plt.plot([d[0] - context.start_ts for d in context.fec.fec_key_data],
             [d[1] for d in context.fec.fec_key_data], '--')
    plt.plot([d[0] - context.start_ts for d in context.fec.fec_delta_data],
             [d[1] for d in context.fec.fec_delta_data], '-.')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('FEC Ratio')
    plt.legend(['Key frame FEC', 'Delta frame FEC'])
    plt.savefig(os.path.join(output_dir, f'set-fec.{FIG_EXTENSION}'), dpi=DPI)


def analyze_stream(context: StreamingContext, output_dir=OUTPUT_DIR) -> None:
    print_statistics(context)
    analyze_frame(context, output_dir)
    analyze_packet(context, output_dir)
    analyze_network(context, output_dir)
    analyze_fec(context, output_dir)


def get_stream_context(result_path=os.path.join(RESULTS_PATH, 'eval_static')) -> StreamingContext:
    sender_log = os.path.join(result_path, 'eval_sender.log')
    context = StreamingContext()
    for line in open(sender_log).readlines():
        try:
            parse_line(line, context)
        except Exception as e:
            print(f"Error parsing line: {line}")
    return context


def main(result_path=os.path.join(RESULTS_PATH, 'eval_static'), sender_log=None) -> None:
    if sender_log is None:
        sender_log = os.path.join(result_path, 'eval_sender.log')
    context = StreamingContext()
    for line in open(sender_log, encoding='utf-8').readlines():
        try:
            parse_line(line, context)
        except Exception as e:
            print(f"Error parsing line: {line}")
    analyze_stream(context, output_dir=result_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--result_path', type=str, default=os.path.join(RESULTS_PATH, 'eval_static'))
    parser.add_argument('-l', '--sender_log', type=str, default=None)
    args = parser.parse_args()
    main(result_path=args.result_path, sender_log=args.sender_log)
