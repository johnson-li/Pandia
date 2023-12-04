from typing import Optional


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

