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

        self.frame_encoding_delay: np.ndarray = np.zeros(
            self.frame_history_size, dtype=np.int32)
        self.frame_transmission_delay: np.ndarray = np.zeros(
            self.frame_history_size, dtype=np.int32)
        self.frame_decoding_delay: np.ndarray = np.zeros(
            self.frame_history_size, dtype=np.int32)
        self.frame_g2g_delay: np.ndarray = np.zeros(
            self.frame_history_size, dtype=np.int32)
        self.frame_size: np.ndarray = np.zeros(self.frame_history_size, dtype=np.int32)
        self.packet_egress_rate: np.ndarray = np.zeros(
            self.packet_history_size, dtype=np.int32)
        self.packet_ack_rate: np.ndarray = np.zeros(
            self.packet_history_size, dtype=np.int32)
        self.pacing_rate: np.ndarray = np.zeros(self.packet_history_size, dtype=np.int32)
        self.pacing_burst_interval: np.ndarray = np.zeros(
            self.packet_history_size, dtype=np.int32)
        self.codec_bitrate: np.ndarray = np.zeros(self.codec_history_size, dtype=np.int32)
        self.codec_fps: np.ndarray = np.zeros(self.codec_history_size, dtype=np.int32)

    def roll(self):
        self.frame_encoding_delay = np.roll(self.frame_encoding_delay, 1)
        self.frame_transmission_delay =  \
            np.roll(self.frame_transmission_delay, 1)
        self.frame_decoding_delay = np.roll(self.frame_decoding_delay, 1)
        self.frame_g2g_delay = np.roll(self.frame_g2g_delay, 1)
        self.frame_size = np.roll(self.frame_size, 1)
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
            [frame.encoding_delay() * 1000 for frame in frames if frame.encoding_delay() >= 0])
        self.frame_transmission_delay[0] = self.calculate_statistics(
            [frame.transmission_delay() * 1000 for frame in frames if frame.transmission_delay() >= 0])
        self.frame_decoding_delay[0] = self.calculate_statistics(
            [frame.decoding_delay() * 1000 for frame in frames if frame.decoding_delay() >= 0])
        self.frame_g2g_delay[0] = self.calculate_statistics(
            [frame.g2g_delay() * 1000 for frame in frames if frame.g2g_delay() >= 0])
        self.frame_size[0] = self.calculate_statistics(
            [frame.encoded_size for frame in frames if frame.encoded_size > 0])
        self.packet_egress_rate[0] = sum([p.size for p in context.latest_egress_packets()])  \
                / self.packet_statistics_duration * 8 / 1024
        self.packet_ack_rate[0] = sum([p.size for p in context.latest_acked_packets()]) \
                / self.packet_statistics_duration * 8 / 1024
        self.pacing_rate[0] = context.networking.pacing_rate_data[-1][1] \
                if context.networking.pacing_rate_data else 0
        self.pacing_burst_interval[0] = context.networking.pacing_burst_interval_data[-1][1] \
                if context.networking.pacing_burst_interval_data else 0
        self.codec_bitrate[0] = context.bitrate_data[-1][1] if context.bitrate_data else 0
        self.codec_fps[0] = context.fps_data[-1][1] if context.fps_data else 0

    def array(self):
        boundary = Observation.boundary()
        keys = sorted(boundary.keys())
        return np.concatenate([getattr(self, k) for k in keys])

    @staticmethod
    def from_array(array):
        observation = Observation()
        boundary = Observation.boundary()
        keys = sorted(boundary.keys())
        for i, k in enumerate(keys):
            setattr(
                observation, k, array[i * DEFAULT_HISTORY_SIZE:(i + 1) * DEFAULT_HISTORY_SIZE])
        return observation

    @staticmethod
    def boundary():
        return {
            'frame_encoding_delay': [0, 1000],
            'frame_transmission_delay': [0, 1000],
            'frame_decoding_delay': [0, 1000],
            'frame_g2g_delay': [0, 1000],
            'frame_size': [0, 1000_000],
            'packet_egress_rate': [0, 500 * 1024 * 1024],
            'packet_ack_rate': [0, 500 * 1024 * 1024],
            'pacing_rate': [0, 500 * 1024 * 1024],
            'pacing_burst_interval': [0, 1000],
            'codec_bitrate': [0, 10 * 1024],
            'codec_fps': [0, 60],
        }

    @staticmethod
    def observation_space():
        boundary = Observation.boundary()
        keys = sorted(boundary.keys())
        low = np.repeat(np.array(
            [boundary[k][0] for k in keys], dtype=np.int32), DEFAULT_HISTORY_SIZE, axis=0)
        high = np.repeat(np.array(
            [boundary[k][1] for k in keys], dtype=np.int32), DEFAULT_HISTORY_SIZE, axis=0)
        return spaces.Box(low=low, high=high, dtype=np.int32)


class Action():
    def __init__(self) -> None:
        self.bitrate = 100
        self.fps = 30
        self.pacing_rate = 100
        self.padding_rate = 100
        self.fec_rate_key = 0
        self.fec_rate_delta = 0
        self.resolution = 720

    def write(self, shm):
        def write_int(value, offset):
            value = int(value)
            bytes = value.to_bytes(4, byteorder='little')
            shm.buf[offset * 4:offset * 4 + 4] = bytes
        write_int(self.bitrate, 0)
        write_int(self.pacing_rate, 1)
        write_int(self.fps, 2)
        write_int(self.fec_rate_key, 3)
        write_int(self.fec_rate_delta, 4)
        write_int(self.padding_rate, 5)

    def array(self):
        boundary = Action.boundary()
        keys = sorted(boundary.keys())
        return np.array([(getattr(self, k) - boundary[k][0]) / (boundary[k][1] - boundary[k][0]) for k in keys])

    @staticmethod
    def from_array(array):
        action = Action()
        boundary = Action.boundary()
        keys = sorted(boundary.keys())
        for i, k in enumerate(keys):
            setattr(action, k, int(float(array[i]) * (boundary[k][1] - boundary[k][0]) + boundary[k][0]))
        if action.bitrate > action.pacing_rate:
            action.pacing_rate = action.bitrate
        return action

    @staticmethod
    def boundary():
        return {
            'bitrate': [10, 10 * 1024],
            # 'fps': [1, 60],
            'pacing_rate': [10, 500 * 1024],
            # 'padding_rate': [0, 500 * 1024],
            # 'fec_rate_key': [0, 255],
            # 'fec_rate_delta': [0, 255],
        }

    @staticmethod
    def action_space():
        boundary = Action.boundary()
        keys = sorted(boundary.keys())
        low = np.zeros(len(keys))
        high = np.ones(len(keys)) 
        return spaces.Box(low=low, high=high)

    @staticmethod
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
        self.observation: Observation = None
        self.observation_space = Observation.observation_space()
        self.action_space = Action.action_space()

    def get_observation(self):
        return self.observation.array()

    def reset(self, *, seed=None, options=None):
        self.step_count = 0
        self.context = StreamingContext()
        self.observation = Observation()
        self.width = 720
        if os.path.isfile(self.sender_log):
            os.remove(self.sender_log)
        try:
            self.shm = shared_memory.SharedMemory(
                name='pandia', create=True, size=Action.shm_size())
            print('Shared memory created: ', self.shm.name)
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(
                name='pandia', create=False, size=Action.shm_size())
            print('Shared memory opened: ', self.shm.name)
        if self.stop_event and not self.stop_event.is_set():
            self.stop_event.set()
        self.stop_event = Event()
        self.monitor_thread = Thread(
            target=monitor_log_file, args=(self.context, self.sender_log, self.stop_event))
        self.monitor_thread.start()
        return self.get_observation()

    def step(self, action):
        print(action)
        action = Action.from_array(action)
        print(f'#{self.step_count} Take action, bitrate: {action.bitrate} kbps, '
              f'pacing rate: {action.pacing_rate} kbps')
        action.write(self.shm)
        if not self.context.codec_initiated:
            assert self.step_count == 0
            start_script = os.path.join(SCRIPTS_PATH, 'start.sh')
            self.p = subprocess.Popen(
                [start_script, '-d', str(self.duration), '-p', '8888', '-w', str(self.width)])
            # Wait for WebRTC SDP negotiation and codec initialization
            print('Waiting for WebRTC...')
            while not self.context.codec_initiated:
                pass
            print('WebRTC is running.')
            self.start_ts = time.time()
        end_ts = self.start_ts + self.step_duration * self.step_count
        sleep_duration = end_ts - time.time()
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        self.observation.append(self.context)
        done = self.p.poll() is not None
        if done:
            self.stop_event.set()
        self.step_count += 1
        reward = self.reward()
        print(f'#{self.step_count} Reward: {reward}')
        return self.get_observation(), reward, done, {}

    def reward(self):
        sla = 100
        factor = 1 if self.observation.frame_g2g_delay[0] <= sla else -1
        quality_score = self.observation.frame_size[0] / 1000
        return factor * quality_score


gym.envs.register(
    id='WebRTCEnv-v0',
    entry_point='pandia.agent.env:WebRTCEnv',
)
