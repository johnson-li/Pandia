import os
from typing import List
from matplotlib import pyplot as plt
import numpy as np
from pandia.agent.utils import divide
from pandia.constants import K
from pandia.context.frame_context import FrameContext
from pandia.context.packet_context import PacketContext
from pandia.context.streaming_context import StreamingContext


FIG_EXTENSION = 'png'
DPI = 600


def illustrate_frame_ts(path: str, context: StreamingContext):
    if not context.frames:
        return
    ids = list(sorted(context.frames.keys()))
    ts_min = context.frames[ids[0]].captured_at_utc
    frames = []
    for frame_id in ids:
        frame = context.frames[frame_id]
        if frame.decoding_at:
            frames.append((frame.captured_at_utc, 
                           frame.encoded_at - frame.captured_at,
                           frame.assembled_at_utc - frame.captured_at_utc,
                           frame.decoded_at_utc - frame.captured_at_utc))
    def plot(i, j):
        x = np.array([f[i] for f in frames])
        y = np.array([f[j] for f in frames])
        indexes = y > 0
        if not np.any(indexes):
            return
        plt.plot(x[indexes] - ts_min, y[indexes] * 1000)

    plt.close()
    for i in range(1, 4):
        plot(0, i)
    plt.ylim(0, 50)
    plt.xlabel('Frame capture time (s)')
    plt.ylabel('Delay (ms)')
    plt.legend(['Encoding Delay', 'Assembly Delay', 'Decoding Delay'])
    plt.savefig(os.path.join(path, 'frame-ts.pdf'))


def illustrate_frame_spec(path: str, context: StreamingContext):
    if not context.frames:
        return
    ids = list(sorted(context.frames.keys()))
    ts_min = context.frames[ids[0]].captured_at_utc
    encoded_size_data = []
    resolution_data = []
    for frame_id in ids:
        frame = context.frames[frame_id]
        if frame.encoded_size > 0:
            encoded_size_data.append((frame.captured_at_utc, frame.encoded_size))
        resolution_data.append((frame.captured_at_utc, frame.height))
    if not encoded_size_data or not resolution_data:
        return
    encoded_size_data = np.array(encoded_size_data)
    resolution_data = np.array(resolution_data)
    plt.close()
    fig, ax1 = plt.subplots()
    ax1.plot(encoded_size_data[:, 0] - ts_min, encoded_size_data[:, 1] / K, 'b')
    ax1.set_xlabel('Frame capture time (s)')
    ax1.set_ylabel('Encoded size (KB)')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    indexes = resolution_data[:, 1] > 0
    ax2.plot(resolution_data[indexes, 0] - ts_min, resolution_data[indexes, 1], 'r')
    ax2.set_ylabel('Resolution (height)')
    ax2.tick_params(axis='y', labelcolor='r')
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'frame-spec.pdf'))


def illustrate_frame_bitrate(path: str, context: StreamingContext):
    if not context.frames:
        return
    ids = list(sorted(context.frames.keys()))
    ts_min = context.frames[ids[0]].captured_at_utc
    bitrate_data = []
    for frame_id in ids:
        frame = context.frames[frame_id]
        if frame.bitrate > 0:
            bitrate_data.append((frame.captured_at_utc, frame.bitrate))
    if not bitrate_data:
        return
    plt.close()
    plt.plot([f[0] - ts_min for f in bitrate_data], [f[1] / K for f in bitrate_data])
    plt.xlabel('Frame capture time (s)')
    plt.ylabel('Bitrate (Kbps)')
    plt.savefig(os.path.join(path, 'frame-bitrate.pdf'))


def analyze_frame(context: StreamingContext, output_dir: str) -> None:
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


def analyze_packet(context: StreamingContext, output_dir: str) -> None:
    data_ack = []
    data_recv = []
    for pkt in sorted(context.packets.values(), key=lambda x: x.sent_at):
        pkt: PacketContext = pkt
        if pkt.sent_at < context.start_ts:
            continue
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
    plt.plot([(d[0] - context.start_ts) for d in data_ack], 
             [d[1] * 1000 for d in data_ack], 'x')
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


def analyze_network(context: StreamingContext, output_dir: str) -> None:
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


def generate_diagrams(path, context):
    os.makedirs(path, exist_ok=True)
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))
    # illustrate_frame_ts(path, context)
    # illustrate_frame_spec(path, context)
    # illustrate_frame_bitrate(path, context)
    analyze_frame(context, path)
    analyze_packet(context, path)
    analyze_network(context, path)
