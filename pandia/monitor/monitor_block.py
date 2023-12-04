from typing import Optional
from pandia.agent.env_config import ENV_CONFIG
from pandia.context.frame_context import FrameContext
from pandia.context.packet_context import PacketContext
from pandia.monitor.monitor_block_data import MonitorBlockData


class MonitorBlock(object):
    def __init__(self, duration=1, boundary=ENV_CONFIG['boundary']) -> None:
        self.ts: float = 0
        self.duration: float = duration
        self.boundary = boundary
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
        self.last_acked_pkt = None
        # Setting statistics
        self.pacing_rate_data = MonitorBlockData(lambda p: p[0], lambda p: p[1], duration=duration)
        self.bitrate_data = MonitorBlockData(lambda f: f.encoding_at, lambda f: f.bitrate, duration=duration)
        # Network statistics
        self.bandwidth_data = MonitorBlockData(lambda p: p[0], lambda p: p[1], duration=duration)

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
    def bandwidth(self):
        return self.bandwidth_data.avg()

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
        return self.frame_encoding_delay_data.avg() if \
            self.frame_encoding_delay_data.num > 0 else \
                self.boundary['frame_encoding_delay'][1]

    @property
    def frame_fps_decoded(self):
        return self.frame_decoded_delay_data.num / self.duration

    @property
    def frame_decoded_delay(self):
        return self.frame_decoded_delay_data.avg() if \
            self.frame_decoded_delay_data.num > 0 else \
                self.boundary['frame_decoded_delay'][1]

    @property
    def frame_decoding_delay(self):
        return self.frame_decoding_delay_data.avg() if \
            self.frame_decoding_delay_data.num > 0 else \
                self.boundary['frame_decoding_delay'][1]

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
        return self.frame_recv_delay_data.avg() if \
            self.frame_recv_delay_data.num > 0 else \
                self.boundary['frame_recv_delay'][1]

    @property
    def frame_egress_delay(self):
        return self.frame_egress_delay_data.avg() if \
            self.frame_egress_delay_data.num > 0 else \
                self.boundary['frame_egress_delay'][1]

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
        # print(f'Packet {packet.rtp_id} sent of {packet.size} bytes, ts: {ts}')
        self.pkts_sent_size.append(packet, ts)
        if packet.last_packet_in_frame and frame:
            self.frame_egress_delay_data.append(frame, ts)

    def on_packet_acked(self, packet: PacketContext, ts: float):
        if packet.received:
            # print(f'Packet {packet.rtp_id} acked of {packet.size} bytes, ts: {ts}')
            self.pkts_acked_size.append(packet, ts)
            self.pkts_trans_delay_data.append(packet, ts)
        if packet.received is not None:
            self.pkts_lost_count.append(packet, ts)
        if self.last_acked_pkt is not None:
            self.pkts_delay_interval_data\
                .append((ts, (packet.received_at_utc - self.last_acked_pkt.received_at_utc) - 
                         (packet.sent_at - self.last_acked_pkt.sent_at)), ts)
        self.last_acked_pkt = packet

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

    def on_bandwidth_updated(self, ts: float, bw: float):
        self.bandwidth_data.append((ts, bw), ts)

    def update_max_queue_delay(self, ts: float):
        pass

