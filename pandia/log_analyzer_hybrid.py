import os
from typing import List

from matplotlib import pyplot as plt

from pandia import DIAGRAMS_PATH, RESULTS_PATH
from pandia.log_analyzer_sender import FrameContext, StreamingContext, parse_line
from pandia.log_analyzer_receiver import Packet, Stream, parse_line as parse_line_receiver


def get_pkt_recv(rtp_id, frame: FrameContext, context_sender: StreamingContext, context_receiver: Stream):
    if rtp_id in frame.retrans_record:
        rtp_id = frame.retrans_record[rtp_id][-1]
    return context_receiver.packets.get(rtp_id, None)


def analyze_frame(frame: FrameContext, context_sender: StreamingContext, context_receiver: Stream) -> tuple:
    recv_ts = []
    rtx_recv_ts = []
    recovery_ts = []
    packets = frame.packets_video()
    for seq_num in range(frame.sequence_range[0], frame.sequence_range[1] + 1):
        if seq_num in context_receiver.packets:
            pkt: Packet = context_receiver.packets[seq_num]
            if seq_num == 30520:
                print(pkt.recv_ts, pkt.recovery_ts)
            if pkt.recovered:
                # The recovered packet is either received by FEC decoding or rtx retransmission
                if pkt.recovery_ts > 0 and (pkt.recovery_ts <= pkt.recv_ts or pkt.recv_ts < 0):
                    recovery_ts.append(pkt.recovery_ts)
                elif pkt.recv_ts > 0 and (pkt.recv_ts <= pkt.recovery_ts or pkt.recovery_ts < 0):
                    rtx_recv_ts.append(pkt.recv_ts)
                else:
                    print(f'WARNING wrong RTP timestamp {seq_num}, recv: {pkt.recv_ts}, recovery: {pkt.recovery_ts}')
            else:
                assert pkt.recv_ts > 0
                recv_ts.append(pkt.recv_ts) 
        else:
            print(f'WARNING RTP {seq_num} from frame {frame.frame_id} not received')
    return recv_ts, rtx_recv_ts, recovery_ts


def summarize(frames: List[FrameContext], context_receiver: Stream):
    for frame in frames:
        if frame.seq_len() <= 0:
            continue
        if frame.packets_video():
            rtp_id_range = [min([p.rtp_id for p in frame.packets_video()]), max([p.rtp_id for p in frame.packets_video()])]
            rtp_sequence_range = [min([p.seq_num for p in frame.packets_video()]), max([p.seq_num for p in frame.packets_video()])]
        else:
            rtp_id_range = [0, 0]
            rtp_sequence_range = [0, 0]
        # print(f'Frame {frame.frame_id} RTP packets, rtp id: {rtp_id_range}, seq num: {frame.sequence_range}')
        assert frame.sequence_range == rtp_sequence_range
    recv_pkts = 0
    recovery_pkts = 0
    retrans_pkts = 0
    for pkt in context_receiver.packets.values():
        if pkt.recovered:
            if pkt.recovery_ts > 0 and (pkt.recovery_ts <= pkt.recv_ts or pkt.recv_ts < 0):
                recovery_pkts += 1
            elif pkt.recv_ts > 0 and (pkt.recv_ts <= pkt.recovery_ts or pkt.recovery_ts < 0):
                retrans_pkts += 1
        else:
            recv_pkts += 1
    all_pkts = recv_pkts + recovery_pkts + retrans_pkts
    print(f'RTP trans report, original: {recv_pkts / all_pkts * 100:.02f}, '
          f'rtx: {retrans_pkts / all_pkts * 100:.02f}, '
          f'recovery: {recovery_pkts / all_pkts * 100:.02f}')

def analyze(output_dir, context_sender: StreamingContext, context_receiver: Stream) -> None:
    frames = list(sorted(context_sender.frames.values(), key=lambda x: x.frame_id))
    video_pacets = [[], []]
    rtx_packets = [[], []]
    recovered_packets = [[], []]
    print(f'Last frame id: {frames[-1].frame_id}')
    for frame in frames[:-2]:  # skip the last frame because it may not be complete
        recv_ts, rtx_recv_ts, recovery_ts = analyze_frame(frame, context_sender, context_receiver)
        if recv_ts:
            video_pacets[0] += [frame.captured_at - context_sender.start_ts] * len(recv_ts)
            video_pacets[1] += [(t - min(recv_ts)) * 1000 for t in recv_ts]
        if rtx_recv_ts:
            rtx_packets[0] += [frame.captured_at - context_sender.start_ts] * len(rtx_recv_ts)
            rtx_packets[1] += [(t - min(recv_ts)) * 1000 for t in rtx_recv_ts]
        if recovery_ts:
            recovered_packets[0] += [frame.captured_at - context_sender.start_ts] * len(recovery_ts)
            recovered_packets[1] += [(t - min(recv_ts)) * 1000 for t in recovery_ts]
    plt.close()
    plt.plot(video_pacets[0], video_pacets[1], '.')
    plt.plot(rtx_packets[0], rtx_packets[1], '8')
    plt.plot(recovered_packets[0], recovered_packets[1], '^')
    plt.xlabel('Frame captured time (s)')
    plt.ylabel('RTP reception time (ms)')
    plt.legend(['RTP', 'Retransmitted RTP', 'FEC recovered RTP'])    
    # plt.ylim([0, 500])
    plt.savefig(os.path.join(output_dir, 'mea-rtp-recv-ts.pdf'))

    frame_completion_data = [[], []]
    for frame in frames[:-2]:
        if not frame.seq_len():
            continue
        recv_count = 0
        for seq_num in range(frame.sequence_range[0], frame.sequence_range[1] + 1):
            if seq_num in context_receiver.packets:
                recv_count += 1
        frame_completion_data[0].append(frame.captured_at - context_sender.start_ts)
        frame_completion_data[1].append(recv_count / frame.seq_len())
    plt.close()
    plt.plot(frame_completion_data[0], frame_completion_data[1], '.')
    plt.xlabel('Frame capture time (s)')
    plt.ylabel('Frame packets reception percentile (%)')
    plt.savefig(os.path.join(output_dir, 'mea-frame-packet-recv-p.pdf'))
    summarize(frames, context_receiver)


def main(result_path=os.path.join(RESULTS_PATH, 'eval_static')):
    sender_log = os.path.join(result_path, 'eval_sender.log')
    receiver_log = os.path.join(result_path, 'eval_receiver.log')
    context = StreamingContext()
    for line in open(sender_log).readlines():
        try:
            parse_line(line, context)
        except Exception as e:
            print(e)
    stream = Stream()
    for line in open(receiver_log).readlines():
        line = line.strip()
        if line:
            try:
                parse_line_receiver(line, stream)
            except Exception as e:
                print(e)
    print("========== STATISTICS [HYBRID] ==========")
    analyze(result_path, context, stream)
    


if __name__ == "__main__":
    main()
