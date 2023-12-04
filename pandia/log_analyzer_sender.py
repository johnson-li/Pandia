import argparse
import os
import re
import time
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from pandia import RESULTS_PATH, DIAGRAMS_PATH
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.normalization import RESOLUTION_LIST
from pandia.analysis.stream_illustrator import analyze_frame, analyze_network, analyze_packet
from pandia.context.frame_context import FrameContext
from pandia.context.packet_context import PacketContext
from pandia.context.streaming_context import StreamingContext


CODEC_NAMES = ['Generic', 'VP8', 'VP9', 'AV1', 'H264', 'Multiplex']
OUTPUT_DIR = DIAGRAMS_PATH
kDeltaTick=.25  # In ms
kBaseTimeTick=kDeltaTick*(1<<8)  # In ms
kTimeWrapPeriod=kBaseTimeTick * (1 << 24)  # In ms
PACKET_TYPES = ['audio', 'video', 'rtx', 'fec', 'padding']
FIG_EXTENSION = 'png'
DPI = 600


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
