import os
import numpy as np
from gymnasium import spaces
from pandia import RESULTS_PATH
from pandia.agent.normalization import NORMALIZATION_RANGE, dnml, nml
from pandia.constants import K, M, G
from typing import Dict, TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from pandia.agent.action import Action
    from pandia.log_analyzer_sender import MonitorBlock


class Observation(object):
    def __init__(self, obs_keys, durations, history_size=1) -> None:
        for obs_key in obs_keys:
            assert obs_key in Observation.boundary(), f'Unknown observation key: {obs_key}'
        self.obs_keys = list(sorted(obs_keys))
        self.history_size = history_size
        self.obs_keys_map = {k: i for i, k in enumerate(self.obs_keys)}
        self.monitor_durations = list(sorted(durations))
        self.data = np.zeros((self.history_size, len(self.monitor_durations), len(self.obs_keys)), dtype=np.float32)

    def roll(self):
        np.roll(self.data, 1, axis=0)

    def __str__(self) -> str:
        duration_index = -1
        data = self.data[0][duration_index]

        def get_data(key):
            res = dnml(key, data[self.obs_keys_map[key]], self.boundary()[key]) if key in self.obs_keys else '?'
            if res != '?' and key in ['frame_bitrate', 'pkt_egress_rate', 'pkt_ack_rate', 'pacing_rate']:
                res = int(res / 1024)
            if res != '?' and key in ['frame_encoding_delay', 'frame_egress_delay', 'frame_recv_delay', 
                                      'frame_decoding_delay', 'frame_decoded_delay', 'pkt_trans_delay']:
                res = int(res * 1000)
            if res != '?' and key in ['frame_size', 'frame_height', 'frame_encoded_height', 'frame_fps', 'frame_qp']:
                res = int(res)
            if res != '?' and key in ['pkt_loss_rate']:
                res = int(res * 100)
            return res

        return f'Dly.f: [{get_data("frame_encoding_delay")}, {get_data("frame_egress_delay")}, '\
               f'{get_data("frame_recv_delay")}, {get_data("frame_decoding_delay")}, {get_data("frame_decoded_delay")}]' \
               f', FPS: {get_data("frame_fps")}' \
               f', size: {get_data("frame_size")} ({get_data("frame_height")}/{get_data("frame_encoded_height")})' \
               f', rates: [{get_data("frame_bitrate")}, {get_data("pkt_egress_rate")}, {get_data("pkt_ack_rate")}, {get_data("pacing_rate")}]' \
               f', QP: {get_data("frame_qp")}' \
               f', Dly.p: {get_data("pkt_trans_delay")} ({get_data("pkt_loss_rate")}%)'

    def append(self, monitor_blocks: Dict[int, 'MonitorBlock'], action: 'Action'):
        self.roll()
        for i, dur in enumerate(self.monitor_durations):
            block = monitor_blocks[dur]
            for j ,obs_key in enumerate(self.obs_keys):
                if obs_key == 'action_gap':
                    data = 0
                else:
                    data = getattr(block, obs_key)
                self.data[0, i, j] = nml(obs_key, np.array([data]), self.boundary()[obs_key])[0]

    def array(self) -> np.ndarray:
        return self.data.flatten()

    @staticmethod
    def from_array(array, obs_keys, durations) -> 'Observation':
        obs = Observation(obs_keys, durations)
        obs.data = array.reshape(obs.data.shape)
        return obs

    @staticmethod
    def boundary() -> Dict[str, List]:
        return {
            'frame_encoding_delay': [0, 1], # s
            'frame_egress_delay': [0, 1], # s
            'frame_recv_delay': [0, 1], # s
            'frame_decoding_delay': [0, 1], # s
            'frame_decoded_delay': [0, 1], # s
            'frame_fps': [0, 60],
            'frame_qp': [0, 255],
            'frame_height': [0, 2160], # pixels
            'frame_encoded_height': [0, 2160], # pixels
            'frame_size': [0, 1000_000], # bytes 
            'frame_bitrate': [0, 100 * M], # bps

            'pkt_egress_rate': [0, 200 * M], # bps
            'pkt_ack_rate': [0, 1000 * 1024 * 1024], # bps
            'pkt_loss_rate': [0, 1], # ratio
            'pkt_trans_delay': [0, 1], # s
            'pkt_delay_interval': [0, 10], 

            'pacing_rate': [0, 200 * M], # bps

            'action_gap': [- 500 * M, 500 * M]
        }

        # # observation used in OnRL
        # if use_OnRL:
        #     return {
        #         'packet_loss_rate': [0, 1],
        #         'packet_delay': [0, 1000],
        #         'packet_delay_interval': [0, 10],
        #         'packet_ack_rate': [0, 500 * 1024 * 1024],
        #         'action_gap': [- 500 * 1024 * 1024, 500 * 1024 * 1024],
        #     }
        # return {
        #     'frame_encoding_delay': [0, 1000],
        #     'frame_pacing_delay': [0, 1000],
        #     'frame_decoding_delay': [0, 1000],
        #     # 'frame_assemble_delay': [0, 1000],
        #     'frame_g2g_delay': [0, 1000],
        #     # 'frame_size': [0, 1000_000],
        #     # 'frame_height': [0, 2160],
        #     # 'frame_encoded_height': [0, 2160],
        #     'frame_bitrate': [0, 100_000],
        #     # 'frame_qp': [0, 255],
        #     # 'codec': [0, 4],
        #     'fps': [0, 60],
        #     'packet_egress_rate': [0, 500 * 1024 * 1024],
        #     'packet_ack_rate': [0, 500 * 1024 * 1024],
        #     'packet_loss_rate': [0, 1],
        #     # 'packet_rtt_mea': [0, 1000],
        #     'packet_delay': [0, 1000],
        #     'packet_delay_interval': [0, 10],
        #     # 'pacing_rate': [0, 500 * 1024 * 1024],
        #     # 'pacing_burst_interval': [0, 1000],
        #     # 'codec_bitrate': [0, 10 * 1024],
        #     # 'codec_fps': [0, 60],
        # }

    def observation_space(self) -> spaces.Box:
        low = np.ones_like(self.data) * NORMALIZATION_RANGE[0]
        high = np.ones_like(self.data) * NORMALIZATION_RANGE[1]
        return spaces.Box(low=low, high=high, dtype=np.float32)


def test(result_path=os.path.join(RESULTS_PATH, 'eval_rllib')):
    obs_keys = Observation.boundary().keys()
    period=1
    monitor_durations = [2]
    obs = Observation(obs_keys, monitor_durations)
    sender_log = os.path.join(result_path, 'eval_sender.log')
    from pandia.log_analyzer_sender import StreamingContext, parse_line
    from pandia.agent.action import Action
    context = StreamingContext(monitor_durations=monitor_durations)
    start_ts = 0
    step = 1
    action = Action(action_keys=Action.boundary().keys())
    for line in open(sender_log).readlines():
        try:
            parse_line(line, context)
            if context.codec_initiated and not start_ts:
                start_ts = context.last_ts
            if start_ts and context.last_ts >= start_ts + step * period:
                obs.append(context.monitor_blocks, action)
                print(f'#{step} {obs}')
                step += 1 
        except Exception as e:
            print(f"Error parsing line: {line}")
            raise e


if __name__ == "__main__":
    test()