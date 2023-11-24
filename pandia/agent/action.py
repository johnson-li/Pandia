from enum import Enum
import numpy as np
from gymnasium import spaces
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.normalization import NORMALIZATION_RANGE, dnml, nml
from pandia.constants import M


class Action():
    def __init__(self, action_keys, boundary=ENV_CONFIG['boundary'], 
                 log=ENV_CONFIG['gym_setting']['log_nml']) -> None:
        self.action_keys = list(sorted(action_keys))
        # Initiation values are invalid values so that WebRTC will not use DRL actions 
        self.bitrate = 0
        self.pacing_rate = 0 
        self.resolution = 0
        self.fps = 0
        self.padding_rate = 0
        self.fec_rate_key = 256
        self.fec_rate_delta = 256
        self.fake = action_keys == ['fake']
        self.boundary = boundary
        self.log = log

    def __str__(self) -> str:
        if self.fake:
            return 'Fake, '
        res = ''
        if 'resolution' in self.action_keys:
            res += f'Res.: {self.resolution}p, '
        if 'pacing_rate' in self.action_keys:
            res += f'P.r.: {self.pacing_rate / M:.02f} mbps, '
        if 'bitrate' in self.action_keys:
            res += f'B.r.: {self.bitrate / M:.02f} mbps, '
        if 'fps' in self.action_keys:
            res += f'FPS: {self.fps}, '
        if 'fec_rate_key' in self.action_keys:
            res += f'FEC: {self.fec_rate_key}/{self.fec_rate_delta}'
        assert res != '', 'Invalid action'
        return res


    def write(self, shm) -> None:
        if type(shm) == bytearray:
            buf = shm
        else:
            buf = shm.buf

        def write_int(value, offset):
            if isinstance(value, np.ndarray):
                value = value[0]
            value = int(value)
            bytes = value.to_bytes(4, byteorder='little')
            buf[offset * 4:offset * 4 + 4] = bytes
        
        parameters = ['bitrate', 'pacing_rate', 'fps', 'fec_rate_key', 
                      'fec_rate_delta', 'padding_rate', 'resolution']
        for i, p in enumerate(parameters):
            if self.fake:
                write_int(ENV_CONFIG['action_invalid_value'][p], i)
            elif p in self.action_keys:
                write_int(getattr(self, p), i)
            else:
                write_int(ENV_CONFIG['action_static_settings'][p], i)

    def array(self) -> np.ndarray:
        keys = sorted(self.action_keys)
        return np.array([nml(k, getattr(self, k), self.boundary[k], log=self.log) for k in keys], dtype=np.float32)

    ## It is usually used for testing purpose. Try to avoid using it in the training process.
    @staticmethod
    def from_array(array: np.ndarray, keys, boundary=ENV_CONFIG['boundary'],
                   log=ENV_CONFIG['gym_setting']['log_nml']) -> 'Action':
        assert array.dtype == np.float32, f'Invalid action array type: {array.dtype}'
        keys = list(sorted(keys))
        action = Action(keys, boundary=boundary)
        for i, k in enumerate(keys):
            setattr(action, k, dnml(k, array[i], boundary[k], log=log))
        parameters = ['bitrate', 'pacing_rate', 'fps', 'fec_rate_key', 
                      'fec_rate_delta', 'padding_rate', 'resolution']
        for p in parameters:
            if p not in keys:
                setattr(action, p, ENV_CONFIG['action_static_settings'][p])

        # Post process to avoid invalid action settings
        if action.bitrate > action.pacing_rate:
            action.pacing_rate = action.bitrate
        return action

    def action_space(self):
        low = np.ones(len(self.action_keys), dtype=np.float32) * NORMALIZATION_RANGE[0]
        high = np.ones(len(self.action_keys), dtype=np.float32) * NORMALIZATION_RANGE[1]
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @staticmethod
    def action_limit(action_keys, boundary=ENV_CONFIG['boundary'], 
                     limit={}, log=ENV_CONFIG['gym_setting']['log_nml']):
        res = np.zeros((2, len(action_keys)), dtype=np.float32)
        for i, key in enumerate(action_keys):
            for j in [0, 1]:
                res[j][i] = nml(key, limit.get(key, boundary[key])[j], 
                                boundary[key], log=log) 
        return spaces.Box(low=res[0], high=res[1], dtype=np.float32)

    @staticmethod
    def shm_size():
        return 10 * 4
