import os
import subprocess
import threading
from threading import Event, Thread
import time
import gym
import numpy as np
from gym import spaces
from multiprocessing import shared_memory
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
        self.frame_statistics_duration = 1
        self.packet_statistics_duration = 1

        self.frame_encoding_delay = np.zeros(self.frame_history_size)
        self.frame_transmission_delay = np.zeros(self.frame_history_size)
        self.frame_decoding_delay = np.zeros(self.frame_history_size)
        self.frame_g2g_delay = np.zeros(self.frame_history_size)
        self.packet_egress_rate = np.zeros(self.packet_history_size)
        self.packet_ack_rate = np.zeros(self.packet_history_size)
        self.pacing_rate = np.zeros(self.packet_history_size)
        self.pacing_burst_interval = np.zeros(self.packet_history_size)
        self.codec_bitrate = np.zeros(self.codec_history_size)
        self.codec_fps = np.zeros(self.codec_history_size)

    def roll(self):
        self.frame_encoding_delay = np.roll(self.frame_encoding_delay, 1)
        self.frame_transmission_delay =  \
            np.roll(self.frame_transmission_delay, 1)
        self.frame_decoding_delay = np.roll(self.frame_decoding_delay, 1)
        self.frame_g2g_delay = np.roll(self.frame_g2g_delay, 1)
        self.packet_egress_rate = np.roll(self.packet_egress_rate, 1)
        self.packet_ack_rate = np.roll(self.packet_ack_rate, 1)
        self.pacing_rate = np.roll(self.pacing_rate, 1)
        self.pacing_burst_interval = np.roll(self.pacing_burst_interval, 1)
        self.codec_bitrate = np.roll(self.codec_bitrate, 1)
        self.codec_fps = np.roll(self.codec_fps, 1)

    def calculate_statistics(self, data):
        if not data:
            return 0
        return np.median(data)

    def append(self, context: StreamingContext):
        self.roll()
        frames = context.latest_frames()
        self.frame_encoding_delay[0] = self.calculate_statistics(
            [frame.encoding_delay() for frame in frames if frame.encoding_delay() >= 0])
        self.frame_transmission_delay[0] = self.calculate_statistics(
            [frame.transmission_delay() for frame in frames if frame.transmission_delay() >= 0])
        self.frame_decoding_delay[0] = self.calculate_statistics(
            [frame.decoding_delay() for frame in frames if frame.decoding_delay() >= 0])
        self.frame_g2g_delay[0] = self.calculate_statistics(
            [frame.g2g_delay() for frame in frames if frame.g2g_delay() >= 0])
        self.packet_egress_rate[0] = sum(
            [p.size for p in context.latest_egress_packets()]) / self.packet_statistics_duration * 8
        self.packet_ack_rate[0] = sum(
            [p.size for p in context.latest_acked_packets()]) / self.packet_statistics_duration * 8
        self.pacing_rate[0] = context.networking.pacing_rate_data[-1][1] if context.networking.pacing_rate_data else 0
        self.pacing_burst_interval[0] = context.networking.pacing_burst_interval_data[-1][1] if context.networking.pacing_burst_interval_data else 0
        self.codec_bitrate[0] = context.bitrate_data[-1][1] if context.bitrate_data else 0
        self.codec_fps[0] = context.fps_data[-1][1] if context.fps_data else 0

    def get(self):
        return {
            'frame_encoding_delay': self.frame_encoding_delay,
            'frame_transmission_delay': self.frame_transmission_delay,
            'frame_decoding_delay': self.frame_decoding_delay,
            'frame_g2g_delay': self.frame_g2g_delay,
            'packet_egress_rate': self.packet_egress_rate,
            'packet_ack_rate': self.packet_ack_rate,
            'pacing_rate': self.pacing_rate,
            'pacing_burst_interval': self.pacing_burst_interval,
            'codec_bitrate': self.codec_bitrate,
            'codec_fps': self.codec_fps,
        }

    def observation_space(self):
        return spaces.Dict({
            'frame_encoding_delay': spaces.Box(low=-1, high=1, shape=(self.frame_history_size,), dtype=np.float32),
            'frame_transmission_delay': spaces.Box(low=-1, high=1, shape=(self.frame_history_size,), dtype=np.float32),
            'frame_decoding_delay': spaces.Box(low=-1, high=1, shape=(self.frame_history_size,), dtype=np.float32),
            'frame_g2g_delay': spaces.Box(low=-1, high=1, shape=(self.frame_history_size,), dtype=np.float32),
            'packet_egress_rate': spaces.Box(low=-1, high=1, shape=(self.packet_history_size,), dtype=np.float32),
            'packet_ack_rate': spaces.Box(low=-1, high=1, shape=(self.packet_history_size,), dtype=np.float32),
            'pacing_rate': spaces.Box(low=-1, high=1, shape=(self.packet_history_size,), dtype=np.float32),
            'pacing_burst_interval': spaces.Box(low=-1, high=1, shape=(self.packet_history_size,), dtype=np.float32),
            'codec_bitrate': spaces.Box(low=-1, high=1, shape=(self.codec_history_size,), dtype=np.float32),
            'codec_fps': spaces.Box(low=-1, high=1, shape=(self.codec_history_size,), dtype=np.float32),
        })


class Action():
    def __init__(self) -> None:
        self.bitrate = 100
        self.fps = 30
        self.pacing_rate = 100
        self.padding_rate = 100

    def shm_size():
        return 10 * 4


class WebRTCEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.sender_log = os.path.join(RESULTS_PATH, 'sender.log')
        self.frame_history_size = 10
        self.packet_history_size = 10
        self.packet_history_duration = 10
        self.step_duration = 1
        self.duration = 30
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
            [start_script, '-d', str(self.duration), '-p', '8888', '-w', str(self.width)])
        try:
            self.shm = shared_memory.SharedMemory(name='pandia', create=True, size=Action.shm_size())
            print('Shared memory created: ', self.shm.name)
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name='pandia', create=False, size=Action.shm_size())
            print('Shared memory opened: ', self.shm.name)
        self.write_action(Action())
        if self.stop_event and not self.stop_event.is_set():
            self.stop_event.set()
        self.stop_event = Event()
        self.monitor_thread = Thread(
            target=monitor_log_file, args=(self.context, self.sender_log, self.stop_event))
        self.monitor_thread.start()
        # Wait for WebRTC SDP negotiation and codec initialization
        while not self.context.codec_initiated:
            pass
        self.start_ts = time.time()
        return self.get_observation(), None

    def write_action(self, action: Action):
        def write_int(value, offset):
            bytes = value.to_bytes(4, byteorder='little')
            self.shm.buf[offset * 4:offset * 4 + 4] = bytes
        write_int(action.bitrate, 0)
        write_int(action.pacing_rate, 1)
        write_int(action.fps, 2)

    def step(self, action):
        action = Action()
        action.bitrate = 1 * 1024
        action.pacing_rate = 500 * 1024
        action.fps = 10
        action.padding_rate = 0
        self.write_action(action)
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
        print(
            f'Step: {env.step_count}, Reward: {reward}, Observation: {observation["frame_g2g_delay"]}')
        if done:
            break


if __name__ == "__main__":
    main()
