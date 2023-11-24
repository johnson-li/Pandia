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
    def __init__(self, obs_keys=ENV_CONFIG['observation_keys'], 
                 durations=ENV_CONFIG['gym_setting']['observation_durations'], 
                 history_size=ENV_CONFIG['gym_setting']['history_size'],
                 boundary=ENV_CONFIG['boundary']) -> None:
        self.obs_keys = list(sorted(obs_keys))
        self.history_size = history_size
        self.obs_keys_map = {k: i for i, k in enumerate(self.obs_keys)}
        self.monitor_durations = list(sorted(durations))
        self.data = np.zeros((self.history_size, len(self.monitor_durations), len(self.obs_keys)), dtype=np.float32)
        self.boundary = boundary

    def roll(self):
        np.roll(self.data, 1, axis=0)

    def get_data(self, data, key, numeric=False):
        res = dnml(key, data[self.obs_keys_map[key]], self.boundary[key], log=False) \
            if key in self.obs_keys else '?'
        if numeric:
            return res
        if res == '?':
            return res
        elif key in ['frame_bitrate', 'pkt_egress_rate', 'pkt_ack_rate', 'pacing_rate', 
                     'bitrate', 'bandwidth']:
            return f'{res / M: .02f}'  # in mbps 
        elif key in ['frame_encoding_delay', 'frame_egress_delay', 'frame_recv_delay', 
                     'frame_decoding_delay', 'frame_decoded_delay', 'pkt_trans_delay',
                     'pkt_delay_interval']:
            return f'{res * 1000: .02f}'  # in ms
        elif res != '?' and key in ['frame_size', 'frame_height', 'frame_encoded_height', 
                                    'frame_fps', 'frame_fps_decoded', 'frame_qp', 'frame_key_count']:
            return f'{int(res)}'
        elif key in ['pkt_loss_rate']:
            return f'{res * 100: .02f}'  # in %
        return res

    def __str__(self) -> str:
        duration_index = 0
        obs_str_list = []
        get_data = self.get_data
        for duration_index in range(len(self.data[0])):
            data = self.data[0][duration_index]
            obs_str = f'Dly.f (ms): [{get_data(data, "frame_encoding_delay")}, {get_data(data, "frame_egress_delay")}, '\
                    f'{get_data(data, "frame_recv_delay")}, {get_data(data, "frame_decoding_delay")}, {get_data(data, "frame_decoded_delay")}]' \
                    f', FPS: {get_data(data, "frame_fps")}/{get_data(data, "frame_fps_decoded")}' \
                    f', size (bytes): {get_data(data, "frame_size")} ({get_data(data, "frame_height")}/{get_data(data, "frame_encoded_height")}, {get_data(data, "frame_key_count")})' \
                    f', rates (mbps): [{get_data(data, "bitrate")}, {get_data(data, "frame_bitrate")}, {get_data(data, "pkt_egress_rate")}, {get_data(data, "pkt_ack_rate")}, {get_data(data, "pacing_rate")}]' \
                    f', bw (mbps): {get_data(data, "bandwidth")}' \
                    f', QP: {get_data(data, "frame_qp")}' \
                    f', Dly.p (ms): [{get_data(data, "pkt_trans_delay")}, {get_data(data, "pkt_delay_interval")}] ({get_data(data, "pkt_loss_rate")}%)'
            obs_str_list.append(obs_str)
        return f'[{", ".join(obs_str_list)}]'

    def append(self, monitor_blocks: Dict[int, 'MonitorBlock'], action: 'Action'):
        self.roll()
        for i, dur in enumerate(self.monitor_durations):
            block = monitor_blocks[dur]
            for j ,obs_key in enumerate(self.obs_keys):
                if obs_key == 'action_gap':
                    data = 0
                else:
                    data = getattr(block, obs_key)
                self.data[0, i, j] = nml(obs_key, np.array([data]), self.boundary[obs_key], log=False)[0]

    def array(self) -> np.ndarray:
        return self.data.flatten()

    @staticmethod
    def from_array(array, obs_keys=ENV_CONFIG['observation_keys'], 
                   durations=ENV_CONFIG['gym_setting']['observation_durations']) -> 'Observation':
        obs = Observation(obs_keys, durations)
        obs.data = array.reshape(obs.data.shape)
        return obs

    def observation_space(self) -> spaces.Box:
        low = np.ones_like(self.data).reshape((-1, )) * NORMALIZATION_RANGE[0]
        high = np.ones_like(self.data).reshape((-1, )) * NORMALIZATION_RANGE[1]
        return spaces.Box(low=low, high=high, dtype=np.float32)
