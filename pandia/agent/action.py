
from enum import Enum
import numpy as np
from gymnasium import spaces

from pandia.agent.normalization import NORMALIZATION_RANGE, dnml, nml


class NML_MODE(Enum):
    DISABLED = 0
    ENABLED = 1


class Action():
    def __init__(self, action_keys) -> None:
        # Initiation values are invalid values so that WebRTC will not use DRL actions 
        self.bitrate = np.array([0, ], dtype=np.int32)
        self.pacing_rate = np.array([0, ], dtype=np.int32)
        self.resolution = np.array([0, ], dtype=np.int32)
        self.fps = np.array([0, ], dtype=np.int32)
        self.padding_rate = np.array([0, ], dtype=np.int32)
        self.fec_rate_key = np.array([256, ], dtype=np.int32)
        self.fec_rate_delta = np.array([256, ], dtype=np.int32)

    @staticmethod
    def boundary() -> dict:
        return {
            'bitrate': [10, 10 * 1024], 
            'fps': [1, 60],
            'pacing_rate': [10, 800 * 1024],
            'padding_rate': [0, 500 * 1024],
            'fec_rate_key': [0, 255],
            'fec_rate_delta': [0, 255],
            'resolution': [0, 1],
        }

    def __str__(self) -> str:
        res = ''
        boundary = Action.boundary()
        if 'resolution' in boundary:
            res += f'Res.: {self.resolution[0]}p, '
        if 'pacing_rate' in boundary:
            res += f'P.r.: {self.pacing_rate[0] / 1024:.02f} mbps, '
        if 'bitrate' in boundary:
            res += f'B.r.: {self.bitrate[0] / 1024:.02f} mbps, '
        if 'fps' in boundary:
            res += f'FPS: {self.fps[0]}, '
        if 'fec_rate_key' in boundary:
            res += f'FEC: {self.fec_rate_key[0]}/{self.fec_rate_delta[0]}'
        return res


    def write(self, shm, no_action=False) -> None:
        def write_int(value, offset):
            if isinstance(value, np.ndarray):
                value = value[0]
            value = int(value)
            bytes = value.to_bytes(4, byteorder='little')
            shm.buf[offset * 4:offset * 4 + 4] = bytes
        write_int(self.bitrate, 0) if not no_action else write_int(0, 0)
        write_int(self.pacing_rate, 1) if not no_action else write_int(0, 1)
        write_int(self.fps, 2) if not no_action else write_int(0, 2)
        write_int(self.fec_rate_key, 3) if not no_action else write_int(256, 3)
        write_int(self.fec_rate_delta, 4) if not no_action else write_int(256, 4)
        write_int(self.padding_rate, 5) if not no_action else write_int(0, 5)
        write_int(self.resolution, 6) if not no_action else write_int(0, 6)

    def array(self) -> np.ndarray:
        boundary = Action.boundary()
        keys = sorted(boundary.keys())
        return np.concatenate([nml(k, getattr(self, k), boundary[k], log=False) for k in keys])

    @staticmethod
    def from_array(array) -> 'Action':
        action = Action()
        boundary = Action.boundary()
        keys = sorted(boundary.keys())
        for i, k in enumerate(keys):
            setattr(action, k, dnml(k, array[i:i+1], boundary[k], log=False))

        # Post process to avoid invalid action settings
        if action.bitrate[0] > action.pacing_rate[0]:
            action.pacing_rate[0] = action.bitrate[0]
        return action

    @staticmethod
    def action_space(legacy_api=False):
        boundary = Action.boundary()
        low = np.ones(len(boundary), dtype=np.float32) * NORMALIZATION_RANGE[0]
        high = np.ones(len(boundary), dtype=np.float32) * NORMALIZATION_RANGE[1]
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @staticmethod
    def shm_size():
        return 10 * 4
