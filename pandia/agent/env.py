import os
import random
import subprocess
import threading
from threading import Event, Thread
import time
from typing import List
import numpy as np
from gymnasium import Env, spaces
from multiprocessing import shared_memory
from pandia import RESULTS_PATH, SCRIPTS_PATH, BIN_PATH
from pandia.log_analyzer import CODEC_NAMES, FrameContext, StreamingContext, parse_line

DEFAULT_HISTORY_SIZE = 3
NORMALIZATION_RANGE = (-1, 1)
RESOLUTION_LIST = [240, 360, 480, 720, 960, 1080, 1440, 2160]


def monitor_webrtc_sender(context: StreamingContext, stdout: str, stop_event: threading.Event, sender_log=None):
    f = open(sender_log, 'w+') if sender_log else None 
    while not stop_event.is_set():
        line = stdout.readline().decode().strip()
        if f:
            f.write(line + '\n')
        if line:
            parse_line(line, context)
    if f:
        f.close()


def normalize(name, value, value_range, normalized_range=NORMALIZATION_RANGE):
    if name == 'resolution':
        return RESOLUTION_LIST.index(min(RESOLUTION_LIST, key=lambda x:abs(x - value))) \
            / len(RESOLUTION_LIST)
    if type(value) == np.ndarray:
        value = np.array(value, dtype=np.float32)
    return (value - value_range[0]) / (value_range[1] - value_range[0]) * \
        (normalized_range[1] - normalized_range[0]) + normalized_range[0]


def denormalize(name, value, value_range, normalized_range=NORMALIZATION_RANGE):
    if name == 'resolution':
        return RESOLUTION_LIST[int(value * len(RESOLUTION_LIST))]
    res = normalize(name, value, normalized_range, value_range)
    if type(res) == np.ndarray:
        res = np.array(res, dtype=np.int32)
    return res


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
        self.frame_pacing_delay: np.ndarray = np.zeros(
            self.frame_history_size, dtype=np.int32)
        self.frame_decoding_delay: np.ndarray = np.zeros(
            self.frame_history_size, dtype=np.int32)
        self.frame_assemble_delay: np.ndarray = np.zeros(
            self.frame_history_size, dtype=np.int32)
        self.frame_g2g_delay: np.ndarray = np.zeros(
            self.frame_history_size, dtype=np.int32)
        self.frame_size: np.ndarray = np.zeros(self.frame_history_size, dtype=np.int32)
        self.frame_height: np.ndarray = np.zeros(self.frame_history_size, dtype=np.int32)
        self.frame_encoded_height: np.ndarray = \
                np.zeros(self.frame_history_size, dtype=np.int32)
        self.frame_bitrate: np.ndarray = np.zeros(self.frame_history_size, dtype=np.int32)
        self.frame_qp: np.ndarray = np.zeros(self.frame_history_size, dtype=np.int32)
        self.codec: np.ndarray = np.zeros(self.frame_history_size, dtype=np.int32)
        self.fps: np.ndarray = np.zeros(self.frame_history_size, dtype=np.int32)
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
        self.frame_pacing_delay = np.roll(self.frame_pacing_delay, 1)
        self.frame_decoding_delay = np.roll(self.frame_decoding_delay, 1)
        self.frame_assemble_delay = np.roll(self.frame_assemble_delay, 1)
        self.frame_g2g_delay = np.roll(self.frame_g2g_delay, 1)
        self.frame_size = np.roll(self.frame_size, 1)
        self.frame_height = np.roll(self.frame_height, 1)
        self.frame_encoded_height = np.roll(self.frame_encoded_height, 1)
        self.frame_bitrate = np.roll(self.frame_bitrate, 1)
        self.frame_qp = np.roll(self.frame_qp, 1)
        self.codec = np.roll(self.codec, 1)
        self.fps = np.roll(self.fps, 1)
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
        frames: List[FrameContext] = context.latest_frames()
        self.frame_encoding_delay[0] = self.calculate_statistics(
            [frame.encoding_delay() * 1000 for frame in frames if frame.encoding_delay() >= 0])
        self.frame_pacing_delay[0] = self.calculate_statistics(
            [frame.pacing_delay() * 1000 for frame in frames if frame.pacing_delay() >= 0])
        self.frame_decoding_delay[0] = self.calculate_statistics(
            [frame.decoding_delay() * 1000 for frame in frames if frame.decoding_delay() >= 0])
        self.frame_assemble_delay[0] = self.calculate_statistics(
            [frame.assemble_delay() * 1000 for frame in frames if frame.assemble_delay() >= 0])
        self.frame_g2g_delay[0] = self.calculate_statistics(
            [frame.g2g_delay() * 1000 for frame in frames if frame.g2g_delay() >= 0])
        self.frame_size[0] = self.calculate_statistics(
            [frame.encoded_size for frame in frames if frame.encoded_size > 0])
        self.frame_height[0] = self.calculate_statistics(
            [frame.height for frame in frames if frame.height > 0])
        self.frame_encoded_height[0] = self.calculate_statistics(
            [frame.encoded_shape[1] for frame in frames if frame.encoded_shape])
        self.frame_bitrate[0] = self.calculate_statistics(
            [frame.bitrate for frame in frames if frame.bitrate > 0])
        self.frame_qp[0] = self.calculate_statistics(
            [frame.qp for frame in frames if frame.qp > 0])
        self.codec[0] = context.codec() 
        self.fps[0] = context.fps()
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
        return np.concatenate([normalize(k, getattr(self, k), boundary[k]) for k in keys])

    @staticmethod
    def from_array(array):
        observation = Observation()
        boundary = Observation.boundary()
        keys = sorted(boundary.keys())
        for i, k in enumerate(keys):
            setattr(observation, k, 
                    denormalize(k, array[i * DEFAULT_HISTORY_SIZE:(i + 1) * DEFAULT_HISTORY_SIZE],    
                                boundary[k]))
        return observation

    @staticmethod
    def boundary():
        return {
            'frame_encoding_delay': [0, 1000],
            'frame_pacing_delay': [0, 1000],
            'frame_decoding_delay': [0, 1000],
            'frame_assemble_delay': [0, 1000],
            'frame_g2g_delay': [0, 1000],
            'frame_size': [0, 1000_000],
            'frame_height': [0, 2160],
            'frame_encoded_height': [0, 2160],
            'frame_bitrate': [0, 100_000],
            'frame_qp': [0, 255],
            'codec': [0, 4],
            'fps': [0, 60],
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
        low = np.ones(len(keys) * DEFAULT_HISTORY_SIZE) * NORMALIZATION_RANGE[0]
        high = np.ones(len(keys) * DEFAULT_HISTORY_SIZE) * NORMALIZATION_RANGE[1]
        return spaces.Box(low=low, high=high, dtype=np.int32)


class Action():
    def __init__(self) -> None:
        self.bitrate = 100
        self.pacing_rate = 100
        self.resolution = 720

        self.fps = 30
        self.padding_rate = 0
        self.fec_rate_key = 0
        self.fec_rate_delta = 0
    
    @staticmethod
    def boundary():
        return {
            'bitrate': [10, 10 * 1024],
            # 'fps': [1, 60],
            'pacing_rate': [10, 500 * 1024],
            # 'padding_rate': [0, 500 * 1024],
            # 'fec_rate_key': [0, 255],
            # 'fec_rate_delta': [0, 255],
            'resolution': [240, 1080],
        }

    def __str__(self) -> str:
        return f'{self.resolution}p, ' \
               f'sending: {self.pacing_rate} / {self.padding_rate} kbps, ' \
               f'{self.bitrate} @ {self.fps} fps, ' \
               f'fec: {self.fec_rate_key} / {self.fec_rate_delta}'

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
        write_int(self.resolution, 6)

    def array(self):
        boundary = Action.boundary()
        keys = sorted(boundary.keys())
        return np.array([normalize(k, getattr(self, k), boundary[k]) for k in keys])

    @staticmethod
    def from_array(array):
        action = Action()
        boundary = Action.boundary()
        keys = sorted(boundary.keys())
        for i, k in enumerate(keys):
            setattr(action, k, int(denormalize(k, array[i], boundary[k])))

        # Post process to avoid invalid action settings
        if action.bitrate > action.pacing_rate:
            action.pacing_rate = action.bitrate
        action.resolution = min(RESOLUTION_LIST, key=lambda x:abs(x - action.resolution))
        return action

    @staticmethod
    def action_space():
        boundary = Action.boundary()
        keys = sorted(boundary.keys())
        low = np.ones(len(keys)) * NORMALIZATION_RANGE[0]
        high = np.ones(len(keys)) * NORMALIZATION_RANGE[1]
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @staticmethod
    def shm_size():
        return 10 * 4


class WebRTCEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        self.uuid = 0
        self.sender_log = config.get('sender_log', None)
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
        self.process_sender = None
        self.process_receiver = None
        self.process_server = None
        self.stop_event = None

    @staticmethod
    def random_uuid():
        while True:
            res = random.randint(9_000, 9_999)
            if os.path.exists(f'/tmp/webrtc_{res}'):
                continue
            with open(f'/tmp/webrtc_{res}', 'w+') as f:
                f.write('')
            return res

    def get_observation(self):
        return self.observation.array()

    def shm_name(self):
        return f'pandia_{self.uuid}'

    def init_webrtc(self):
        try:
            self.shm = shared_memory.SharedMemory(
                name=self.shm_name(), create=True, size=Action.shm_size())
            print('Shared memory created: ', self.shm.name)
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(
                name=self.shm_name(), create=False, size=Action.shm_size())
            print('Shared memory opened: ', self.shm.name)
        self.stop_event = Event()
        self.process_server = subprocess.Popen([os.path.join(BIN_PATH, 'peerconnection_server'), '--port', str(self.uuid)], 
                                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False)
        self.process_receiver = subprocess.Popen([os.path.join(BIN_PATH, 'peerconnection_client_headless'), 
                                                  '--port', str(self.uuid), '--name', 'receiver', 
                                                  '--receiving_only', 'true', 
                                                  '--force_fieldtrials=WebRTC-FlexFEC-03-Advertised/Enabled/WebRTC-FlexFEC-03/Enabled/'], 
                                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False)

    def start_webrtc(self):
        self.process_sender = subprocess.Popen([os.path.join(BIN_PATH, 'peerconnection_client_headless'), 
                                                '--port', str(self.uuid), '--name', 'sender',
                                                '--width', str(self.width), '--fps', str(30), '--autocall', 'true',
                                                '--force_fieldtrials=WebRTC-FlexFEC-03-Advertised/Enabled/WebRTC-FlexFEC-03/Enabled/'], 
                                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        stdout = self.process_sender.stderr
        self.monitor_thread = Thread(
            target=monitor_webrtc_sender, args=(self.context, stdout, self.stop_event, self.sender_log))
        self.monitor_thread.start()

    def reset(self, *, seed=None, options=None):
        if self.process_sender:
            self.process_sender.kill()
        if self.process_receiver:
            self.process_receiver.kill()
        if self.process_server:
            self.process_server.kill()
        if self.stop_event and not self.stop_event.isSet():
            self.stop_event.set()
        self.uuid = self.random_uuid()
        self.step_count = 0
        self.context = StreamingContext()
        self.observation = Observation()
        self.width = 2160
        self.init_webrtc()
        return self.get_observation(), {}
    
    def step(self, action):
        action = Action.from_array(action)
        # print(f'#{self.step_count} Take action: {action}')
        action.write(self.shm)
        if not self.context.codec_initiated:
            assert self.step_count == 0
            self.start_webrtc()
            print('Waiting for WebRTC ready...')
            while not self.context.codec_initiated:
                pass
            print('WebRTC is running.')
            self.start_ts = time.time()
        end_ts = self.start_ts + self.step_duration * self.step_count
        sleep_duration = end_ts - time.time()
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        self.observation.append(self.context)
        done = self.process_sender.poll() is not None or self.process_receiver.poll() is not None
        self.step_count += 1
        reward = self.reward()
        # print(f'#{self.step_count} Reward: {reward}')
        return self.get_observation(), reward, False, done, {}

    def close(self):
        self.stop_event.set()
        self.process_sender.kill()
        self.process_receiver.kill()
        self.process_server.kill()
        self.shm.close()
        self.shm.unlink()

    def reward(self):
        if self.observation.fps[0] == 0:
            return -10
        delay_score = self.observation.frame_g2g_delay[0] / 100
        quality_score = self.observation.frame_bitrate[0] / 1000
        return quality_score - delay_score


# gym.envs.register(
#     id='WebRTCEnv-v0',
#     entry_point='pandia.agent.env:WebRTCEnv',
# )
