
from typing import List

from pandia.context.packet_context import PacketContext


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

    def encoding_delay0(self):
        return self.encoded_at - self.encoding_at if self.encoded_at > 0 else -1

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

    def decoding_delay0(self, utc_offset=.0):
        return self.decoded_at - self.decoding_at if self.decoded_at > 0 else -1

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
