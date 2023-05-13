import os
import subprocess
import threading
from threading import Thread
import time
import gym
import numpy as np
from gym import spaces
from pandia import RESULTS_PATH, SCRIPTS_PATH
from pandia.log_analyzer import StreamingContext, parse_line

log_dir = os.path.join(os.path.dirname(__file__), 'log')


def monitor_log_file(context: StreamingContext, log_path: str):
    def follow(f):
        f.seek(0, os.SEEK_END)
    
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
            yield line

    f = None
    while not f:
        if not f:
            try:
                f = open(log_path)
            except FileNotFoundError:
                time.sleep(0.1)
        
    for line in follow(f):
        line = line.strip()
        if line:
            parse_line(line, context)


class WebRTCEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.sender_log = os.path.join(RESULTS_PATH, 'sender.log')
        self.frame_history_size = 10
        self.packet_history_size = 10
        self.packet_history_duration = 10
        self.step_duration = .1
        self.start_ts = time.time()
        self.step_count = 0
        self.monitor_thread: Thread = None
        self.context: StreamingContext = None
        self.observation_space = spaces.Dict({
            'frame_encoding_delay': spaces.Box(low=-1, high=1, shape=(self.frame_history_size,), dtype=np.float32),
            'frame_transmission_delay': spaces.Box(low=-1, high=1, shape=(self.frame_history_size,), dtype=np.float32),
            'frame_decoding_delay': spaces.Box(low=-1, high=1, shape=(self.frame_history_size,), dtype=np.float32),
            'frame_ack_completeness': spaces.Box(low=-1, high=1, shape=(self.frame_history_size,), dtype=np.float32),
            'frame_drop': spaces.Discrete(self.frame_history_size),
            'packet_egress_rate': spaces.Box(low=-1, high=1, shape=(self.frame_history_size,), dtype=np.float32),
            'packet_ack_rate': spaces.Box(low=-1, high=1, shape=(self.frame_history_size,), dtype=np.float32),
            'pacing_rate': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
        })
        self.action_space = spaces.Dict({
            'target_bitrate': spaces.Box(low=-1, high=1, shape=(self.packet_history_size,), dtype=np.float32),
            'fps': spaces.Box(low=-1, high=1, shape=(self.packet_history_size,), dtype=np.float32),
            'pacing_rate': spaces.Box(low=-1, high=1, shape=(self.packet_history_size,), dtype=np.float32),
            'resolution_width': spaces.Box(low=-1, high=1, shape=(self.packet_history_size,), dtype=np.float32),
        })

    def get_observation(self):
        if self.context.frames:
            print(f'Latest frame id: {max(self.context.frames.keys())}')
        return None

    def reset(self, *, seed=None, options=None):
        self.context = StreamingContext()
        self.width = 720
        if os.path.isfile(self.sender_log):
            os.remove(self.sender_log)
        start_script = os.path.join(SCRIPTS_PATH, 'start.sh')
        self.p = subprocess.Popen(
            [start_script, '-d', '30', '-p', '8888', '-w', str(self.width)])
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.terminate()
        self.monitor_thread = Thread(
            target=monitor_log_file, args=(self.context, self.sender_log, ))
        self.monitor_thread.start()
        return None, None

    def step(self, action):
        self.step_count += 1
        end_ts = self.start_ts + self.step_duration * self.step_count
        sleep_duration = end_ts - time.time()
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        done = self.p.returncode is not None
        if done:
            self.monitor_thread.terminate()
        return self.get_observation(), self.reward(), done, None

    def reward(self):
        return 0


def main():
    env = WebRTCEnv()
    observation, info = env.reset(seed=42)
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(f'Step: {env.step_count}, Reward: {reward}')
        if done:
            break


if __name__ == "__main__":
    main()
