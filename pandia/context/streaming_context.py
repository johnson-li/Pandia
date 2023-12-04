
from typing import Dict, Optional
from pandia.context.action_context import ActionContext
from pandia.context.fec_context import FecContext
from pandia.context.frame_context import FrameContext
from pandia.context.network_context import NetworkContext
from pandia.context.packet_context import PacketContext
from pandia.monitor.monitor_block import MonitorBlock


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

