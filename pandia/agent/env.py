import os
import gym
import numpy as np
from gym import spaces
from pandia.log_analyzer import StreamingContext

log_dir = os.path.join(os.path.dirname(__file__), 'log')


class WebRTCEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        sender_log = os.path.join(log_dir, 'sender.log')
        context = StreamingContext()
        self.history_size = 10
        self.observation_space = spaces.Dict({
            'frame_encoding_delay': spaces.Box(low=-1, high=1, shape=(self.history_size,), dtype=np.float32), 
            'frame_transmission_delay': spaces.Box(low=-1, high=1, shape=(self.history_size,), dtype=np.float32),
            'frame_decoding_delay': spaces.Box(low=-1, high=1, shape=(self.history_size,), dtype=np.float32),
            'frame_loss': spaces.Discrete(self.history_size),
            'packet_egress_rate': spaces.Box(low=-1, high=1, shape=(self.history_size,), dtype=np.float32),
            'packet_ack_rate': spaces.Box(low=-1, high=1, shape=(self.history_size,), dtype=np.float32),
            'pacing_rate': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
        })
        self.action_space = spaces.Dict({
            'target_bitrate': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'fps': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'pacing_rate': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'resolution_width': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
        })


def main():
    pass


if __name__ == "__main__":
    main()
