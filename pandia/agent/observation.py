import os
import numpy as np
from gymnasium import spaces
from pandia import RESULTS_PATH
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.normalization import NORMALIZATION_RANGE, dnml, nml
from pandia.constants import K, M, G
from pandia.agent.action import Action
from typing import Dict, TYPE_CHECKING, List


if TYPE_CHECKING:
    from pandia.log_analyzer_sender import MonitorBlock


ONRL_OBS_KEYS = ['pkt_loss_rate', 'pkt_trans_delay', 'pkt_delay_interval', 'pkt_ack_rate', 'action_gap']

class Observation(object):
    def __init__(self, obs_keys=ENV_CONFIG['observation_keys'], durations=ENV_CONFIG['observation_durations'], 
                 history_size=ENV_CONFIG['history_size']) -> None:
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
        duration_index = 0
        data = self.data[0][duration_index]

        def get_data(key):
            res = dnml(key, data[self.obs_keys_map[key]], self.boundary()[key], log=False) if key in self.obs_keys else '?'
            if res != '?' and key in ['frame_bitrate', 'pkt_egress_rate', 'pkt_ack_rate', 'pacing_rate', 'bitrate']:
                res = int(res / 1024)
            if res != '?' and key in ['frame_encoding_delay', 'frame_egress_delay', 'frame_recv_delay', 
                                      'frame_decoding_delay', 'frame_decoded_delay', 'pkt_trans_delay',
                                      'pkt_delay_interval']:
                res = int(res * 1000)
            if res != '?' and key in ['frame_size', 'frame_height', 'frame_encoded_height', 
                                      'frame_fps', 'frame_fps_decoded', 'frame_qp', 'frame_key_count']:
                res = int(res)
            if res != '?' and key in ['pkt_loss_rate']:
                res = int(res * 100)
            return res

        return f'Dly.f: [{get_data("frame_encoding_delay")}, {get_data("frame_egress_delay")}, '\
               f'{get_data("frame_recv_delay")}, {get_data("frame_decoding_delay")}, {get_data("frame_decoded_delay")}]' \
               f', FPS: {get_data("frame_fps")}/{get_data("frame_fps_decoded")}' \
               f', size: {get_data("frame_size")} ({get_data("frame_height")}/{get_data("frame_encoded_height")}, {get_data("frame_key_count")})' \
               f', rates: [{get_data("bitrate")}, {get_data("frame_bitrate")}, {get_data("pkt_egress_rate")}, {get_data("pkt_ack_rate")}, {get_data("pacing_rate")}]' \
               f', QP: {get_data("frame_qp")}' \
               f', Dly.p: [{get_data("pkt_trans_delay")}, {get_data("pkt_delay_interval")}] ({get_data("pkt_loss_rate")}%)'

    def append(self, monitor_blocks: Dict[int, 'MonitorBlock'], action: 'Action'):
        self.roll()
        for i, dur in enumerate(self.monitor_durations):
            block = monitor_blocks[dur]
            for j ,obs_key in enumerate(self.obs_keys):
                if obs_key == 'action_gap':
                    data = 0
                else:
                    data = getattr(block, obs_key)
                self.data[0, i, j] = nml(obs_key, np.array([data]), self.boundary()[obs_key], log=False)[0]

    def array(self) -> np.ndarray:
        return self.data.flatten()

    @staticmethod
    def from_array(array, obs_keys, durations) -> 'Observation':
        obs = Observation(obs_keys, durations)
        obs.data = array.reshape(obs.data.shape)
        return obs

    @staticmethod
    def boundary() -> Dict[str, List]:
        fps_range = [0, 60] # frames per second
        delay_range = [0, 1] # s
        qp_range = [0, 255]
        resolution_range = [0, 2160]
        return {
            'frame_encoding_delay': delay_range, # s
            'frame_egress_delay': delay_range, # s
            'frame_recv_delay': delay_range, # s
            'frame_decoding_delay': delay_range, # s
            'frame_decoded_delay': delay_range, # s
            'frame_fps': fps_range,
            'frame_fps_decoded': fps_range,
            'frame_qp': qp_range,
            'frame_height': resolution_range, # pixels
            'frame_encoded_height': resolution_range, # pixels
            'frame_size': [0, 1000_000], # bytes 
            'frame_bitrate': [0, 4 * M], # bps
            'frame_key_count': [0, 10], 

            'pkt_egress_rate': [0, 200 * M], # bps
            'pkt_ack_rate': [0, 1000 * M], # bps
            'pkt_loss_rate': [0, 1], # ratio
            'pkt_trans_delay': delay_range, # s
            'pkt_delay_interval': [0, 10], 

            'pacing_rate': [0, 200 * M], # bps

            'action_gap': [- 100 * M, 100 * M],

            # The value is the same as the action, so reuse the action boundary. 
            # Converting from kbps to bps
            'bitrate': [b * 1024 for b in Action.boundary()['bitrate']],  # bps
        }

    def observation_space(self) -> spaces.Box:
        low = np.ones_like(self.data).reshape((-1, )) * NORMALIZATION_RANGE[0]
        high = np.ones_like(self.data).reshape((-1, )) * NORMALIZATION_RANGE[1]
        return spaces.Box(low=low, high=high, dtype=np.float32)


def test(result_path=os.path.join(RESULTS_PATH, 'eval_rllib')):
    obs_keys = Observation.boundary().keys()
    period = 1
    monitor_durations = [1]
    obs = Observation(obs_keys, monitor_durations)
    sender_log = os.path.join(result_path, 'eval_sender.log')
    from pandia.log_analyzer_sender import StreamingContext, parse_line
    from pandia.agent.action import Action
    context = StreamingContext(monitor_durations=monitor_durations)
    from pandia.ntp.ntpclient import NTP_OFFSET_PATH
    if os.path.isfile(NTP_OFFSET_PATH):
        data = open(NTP_OFFSET_PATH, 'r').read().split(',')
        context.update_utc_offset(float(data[0]))
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