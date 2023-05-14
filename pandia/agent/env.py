import os
import subprocess
import threading
from threading import Event, Thread
import time
import gym
import numpy as np
from gym import spaces
from pandia import RESULTS_PATH, SCRIPTS_PATH
from pandia.log_analyzer import StreamingContext, parse_line

DEFAULT_HISTORY_SIZE = 3

def monitor_log_file(context: StreamingContext, log_path: str, stop_event: threading.Event):
    def follow(f):
        f.seek(0, os.SEEK_END)

        while not stop_event.is_set():
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


class Observation(object):
    def __init__(self, frame_history_size=DEFAULT_HISTORY_SIZE, 
                 packet_history_size=DEFAULT_HISTORY_SIZE, 
                 codec_history_size=DEFAULT_HISTORY_SIZE) -> None:
        self.frame_history_size = frame_history_size
        self.packet_history_size = packet_history_size
        self.codec_history_size = codec_history_size

        self.frame_encoding_delay = np.zeros(self.frame_history_size)
        self.frame_transmission_delay = np.zeros(self.frame_history_size)
        self.frame_decoding_delay = np.zeros(self.frame_history_size)
        self.frame_ack_completeness = np.zeros(self.frame_history_size)
        self.frame_drop = np.zeros(self.frame_history_size)
        self.packet_egress_rate = np.zeros(self.packet_history_size)
        self.packet_ack_rate = np.zeros(self.packet_history_size)
        self.pacing_rate = np.zeros(self.packet_history_size)
        self.codec_bitrate = np.zeros(self.codec_history_size)
        self.codec_fps = np.zeros(self.codec_history_size)

    def append(self, context: StreamingContext):
        self.frame_encoding_delay = np.roll(self.frame_encoding_delay, 1)
        self.frame_transmission_delay = np.roll(self.frame_transmission_delay, 1)
        self.frame_decoding_delay = np.roll(self.frame_decoding_delay, 1)
        self.frame_ack_completeness = np.roll(self.frame_ack_completeness, 1)
        self.frame_drop = np.roll(self.frame_drop, 1)
        self.packet_egress_rate = np.roll(self.packet_egress_rate, 1)
        self.packet_ack_rate = np.roll(self.packet_ack_rate, 1)
        self.pacing_rate = np.roll(self.pacing_rate, 1)
        self.codec_bitrate = np.roll(self.codec_bitrate, 1)
        self.codec_fps = np.roll(self.codec_fps, 1)
        self.frame_encoding_delay[0] = 0
        self.frame_transmission_delay[0] = 0
        self.frame_decoding_delay[0] = 0
        self.frame_ack_completeness[0] = 0
        self.frame_drop[0] = 0
        self.packet_egress_rate[0] = 0
        self.packet_ack_rate[0] = 0
        self.pacing_rate[0] = 0
        self.codec_bitrate[0] = context.bitrate_data[-1][1] if context.bitrate_data else 0
        self.codec_fps[0] = context.fps_data[-1][1] if context.fps_data else 0

    def get(self):
        return {
            'frame_encoding_delay': self.frame_encoding_delay,
            'frame_transmission_delay': self.frame_transmission_delay,
            'frame_decoding_delay': self.frame_decoding_delay,
            'frame_ack_completeness': self.frame_ack_completeness,
            'frame_drop': self.frame_drop,
            'packet_egress_rate': self.packet_egress_rate,
            'packet_ack_rate': self.packet_ack_rate,
            'pacing_rate': self.pacing_rate,
            'codec_bitrate': self.codec_bitrate,
            'codec_fps': self.codec_fps,
        }

    def observation_space(self):
        return spaces.Dict({
            'frame_encoding_delay': spaces.Box(low=-1, high=1, shape=(self.frame_history_size,), dtype=np.float32),
            'frame_transmission_delay': spaces.Box(low=-1, high=1, shape=(self.frame_history_size,), dtype=np.float32),
            'frame_decoding_delay': spaces.Box(low=-1, high=1, shape=(self.frame_history_size,), dtype=np.float32),
            'frame_ack_completeness': spaces.Box(low=-1, high=1, shape=(self.frame_history_size,), dtype=np.float32),
            'frame_drop': spaces.Discrete(self.frame_history_size),
            'packet_egress_rate': spaces.Box(low=-1, high=1, shape=(self.packet_history_size,), dtype=np.float32),
            'packet_ack_rate': spaces.Box(low=-1, high=1, shape=(self.packet_history_size,), dtype=np.float32),
            'pacing_rate': spaces.Box(low=-1, high=1, shape=(self.packet_history_size,), dtype=np.float32),
            'codec_bitrate': spaces.Box(low=-1, high=1, shape=(self.codec_history_size,), dtype=np.float32),
            'codec_fps': spaces.Box(low=-1, high=1, shape=(self.codec_history_size,), dtype=np.float32),
        })



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
        self.stop_event: Event = None
        self.context: StreamingContext = None
        self.observation = Observation()
        self.observation_space = self.observation.observation_space()
        self.action_space = spaces.Dict({
            'target_bitrate': spaces.Box(low=-1, high=1, shape=(self.packet_history_size,), dtype=np.float32),
            'fps': spaces.Box(low=-1, high=1, shape=(self.packet_history_size,), dtype=np.float32),
            'pacing_rate': spaces.Box(low=-1, high=1, shape=(self.packet_history_size,), dtype=np.float32),
            'resolution_width': spaces.Box(low=-1, high=1, shape=(self.packet_history_size,), dtype=np.float32),
        })

    def get_observation(self):
        return self.observation.get()

    def reset(self, *, seed=None, options=None):
        self.context = StreamingContext()
        self.observation = Observation()
        self.width = 720
        if os.path.isfile(self.sender_log):
            os.remove(self.sender_log)
        start_script = os.path.join(SCRIPTS_PATH, 'start.sh')
        self.p = subprocess.Popen(
            [start_script, '-d', '30', '-p', '8888', '-w', str(self.width)])
        if self.stop_event and not self.stop_event.is_set():
            self.stop_event.set()
        self.stop_event = Event()
        self.monitor_thread = Thread(
            target=monitor_log_file, args=(self.context, self.sender_log, self.stop_event))
        self.monitor_thread.start()
        return None, None

    def step(self, action):
        self.step_count += 1
        end_ts = self.start_ts + self.step_duration * self.step_count
        sleep_duration = end_ts - time.time()
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        self.observation.append(self.context)
        done = self.p.poll() is not None
        if done:
            self.stop_event.set()
        return self.get_observation(), self.reward(), done, None

    def reward(self):
        return 0


def main():
    env = WebRTCEnv()
    observation, info = env.reset(seed=42)
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(f'Step: {env.step_count}, Reward: {reward}, Observation: {observation}')
        if done:
            break


if __name__ == "__main__":
    main()
