import os

from matplotlib import pyplot as plt

from pandia import DIAGRAMS_PATH, RESULTS_PATH
from pandia.log_analyzer import FrameContext, StreamingContext, parse_line
from pandia.log_analyzer_receiver import Stream, parse_line as parse_line_receiver


def get_pkt_recv(rtp_id, frame: FrameContext, context_sender: StreamingContext, context_receiver: Stream):
    if rtp_id in frame.retrans_record:
        rtp_id = frame.retrans_record[rtp_id][-1]
    return context_receiver.packets.get(rtp_id, None)


def analyze_frame(frame: FrameContext, context_sender: StreamingContext, context_receiver: Stream) -> tuple:
    recv_ts = []
    rtx_recv_ts = []
    packets = frame.packets_video()
    rtp_pkts_num = len(packets)
    for pkt in packets:
        pkt_recv = context_receiver.packets.get(pkt.rtp_id, None)
        if not pkt_recv:
            rtx_packets = [context_receiver.packets[i] for i in frame.retrans_record.get(pkt.rtp_id, []) if i in context_receiver.packets]
            if rtx_packets:
                pkt_recv = rtx_packets[0]
        if pkt_recv:
            if pkt_recv.rtp_id == pkt.rtp_id:
                recv_ts.append(pkt_recv.recv_ts)
            else:
                rtx_recv_ts.append(pkt_recv.recv_ts)
        else:
            print(f'WARNING {pkt.rtp_id} not received nor retransmitted, frame {frame.frame_id}')
    return recv_ts, rtx_recv_ts


def analyze(output_dir, context_sender: StreamingContext, context_receiver: Stream) -> None:
    frames = list(sorted(context_sender.frames.values(), key=lambda x: x.frame_id))
    data1 = [[], []]
    data2 = [[], []]
    print(f'Last frame id: {frames[-1].frame_id}')
    for frame in frames[:-2]:  # skip the last frame because it may not be complete
        recv_ts, rtx_recv_ts = analyze_frame(frame, context_sender, context_receiver)
        if recv_ts:
            data1[0] += [frame.captured_at - context_sender.start_ts] * len(recv_ts)
            data1[1] += [(t - min(recv_ts)) * 1000 for t in recv_ts]
        if rtx_recv_ts:
            data2[0] += [frame.captured_at - context_sender.start_ts] * len(rtx_recv_ts)
            data2[1] += [(t - min(recv_ts)) * 1000 for t in rtx_recv_ts]
    plt.close()
    plt.plot(data1[0], data1[1], '.')
    plt.plot(data2[0], data2[1], '.')
    plt.xlabel('Frame captured time (s)')
    plt.ylabel('RTP reception time (ms)')
    plt.legend(['RTP', 'Retransmitted RTP'])    
    plt.ylim([0, 500])
    plt.savefig(os.path.join(output_dir, 'mea-rtp-recv-ts.pdf'))


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
    analyze(result_path, context, stream)
    


if __name__ == "__main__":
    main()
