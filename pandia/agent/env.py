import os
import subprocess
import threading
from threading import Event, Thread
import time
from typing import List
import numpy as np
import gymnasium
from gymnasium import Env, spaces
from multiprocessing import shared_memory, current_process
from pandia import RESULTS_PATH, SCRIPTS_PATH, BIN_PATH
from pandia.log_analyzer_sender import CODEC_NAMES, ActionContext, FrameContext, StreamingContext, parse_line
from pandia.agent.normalization import nml, dnml, NORMALIZATION_RANGE
from gym.spaces.box import Box
import gym


DEFAULT_HISTORY_SIZE = 3
use_OnRL=True


def log(s: str):
    print(f'[{current_process().pid}] {s}')


def monitor_webrtc_sender(context: StreamingContext, stdout: str, stop_event: threading.Event, sender_log=None):
    if sender_log and os.path.exists(sender_log):
        os.remove(sender_log)
    f = open(sender_log, 'w+') if sender_log else None
    while not stop_event.is_set():
        line = stdout.readline().decode().strip()
        if line:
            if f:
                f.write(line + '\n')
            parse_line(line, context)
            # if not context.codec_initiated:
            #     log(line)
    if f:
        f.close()


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
        self.packet_loss_rate: np.ndarray = np.zeros(
            self.packet_history_size, dtype=np.int32)
        self.packet_rtt_mea: np.ndarray = np.zeros(
            self.packet_history_size, dtype=np.int32)
        self.packet_delay: np.ndarray = np.zeros(self.packet_history_size, dtype=np.int32)
        self.packet_delay_interval: np.ndarray = np.zeros(self.packet_history_size, dtype=np.int32)
        self.pacing_rate: np.ndarray = np.zeros(self.packet_history_size, dtype=np.int32)
        self.pacing_burst_interval: np.ndarray = np.zeros(
            self.packet_history_size, dtype=np.int32)
        self.codec_bitrate: np.ndarray = np.zeros(self.codec_history_size, dtype=np.int32)
        self.codec_fps: np.ndarray = np.zeros(self.codec_history_size, dtype=np.int32)
        self.action_gap: np.ndarray = np.zeros(self.codec_history_size, dtype=np.int32)

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
        self.packet_loss_rate = np.roll(self.packet_loss_rate, 1)
        self.packet_rtt_mea = np.roll(self.packet_rtt_mea, 1)
        self.packet_delay = np.roll(self.packet_delay, 1)
        self.packet_delay_interval = np.roll(self.packet_delay_interval, 1)
        self.pacing_rate = np.roll(self.pacing_rate, 1)
        self.pacing_burst_interval = np.roll(self.pacing_burst_interval, 1)
        self.codec_bitrate = np.roll(self.codec_bitrate, 1)
        self.codec_fps = np.roll(self.codec_fps, 1)
        self.action_gap = np.roll(self.action_gap, 1)

    def __str__(self) -> str:
        delays = (self.frame_encoding_delay[0],
                  self.frame_pacing_delay[0],
                  self.frame_assemble_delay[0],
                  self.frame_g2g_delay[0])
        return f'Dly.: {delays}, ' \
               f'{self.frame_height[0]}p/{self.frame_encoded_height[0]}p, ' \
               f'FPS: {self.fps[0]}, ' \
               f'Codec: {CODEC_NAMES[self.codec[0]]}, ' \
               f'size: {self.frame_size[0]} bytes, ' \
               f'B.r.: {self.frame_bitrate[0]} kbps, ' \
               f'QP: {self.frame_qp[0]}, '

    def calculate_statistics(self, data):
        if not data:
            return -1
        return np.median(data)

    def append(self, context: StreamingContext, action: "Action") -> None:
        self.roll()
        frames: List[FrameContext] = context.latest_frames()
        packets: List[PacketContext] = context.latest_packets()
        self.frame_encoding_delay[0] = self.calculate_statistics(
            [frame.encoding_delay() * 1000 for frame in frames if frame.encoding_delay() >= 0])
        if self.frame_encoding_delay[0] < 0 and 'frame_encoding_delay' in self.boundary():
            self.frame_encoding_delay[0] = self.boundary()['frame_encoding_delay'][1]
        self.frame_pacing_delay[0] = self.calculate_statistics(
            [frame.pacing_delay() * 1000 for frame in frames if frame.pacing_delay() >= 0])
        if self.frame_pacing_delay[0] < 0 and 'frame_pacing_delay' in self.boundary():
            self.frame_pacing_delay[0] = self.boundary()['frame_pacing_delay'][1]
        self.frame_decoding_delay[0] = self.calculate_statistics(
            [frame.decoding_delay() * 1000 for frame in frames if frame.decoding_delay() >= 0])
        if self.frame_decoding_delay[0] < 0 and 'frame_decoding_delay' in self.boundary():
            self.frame_decoding_delay[0] = self.boundary()['frame_decoding_delay'][1]
        self.frame_assemble_delay[0] = self.calculate_statistics(
            [frame.assemble_delay() * 1000 for frame in frames if frame.assemble_delay() >= 0])
        if self.frame_assemble_delay[0] < 0 and 'frame_assemble_delay' in self.boundary():
            self.frame_assemble_delay[0] = self.boundary()['frame_assemble_delay'][1]
        self.frame_g2g_delay[0] = self.calculate_statistics(
            [frame.g2g_delay() * 1000 for frame in frames if frame.g2g_delay() >= 0])
        if self.frame_g2g_delay[0] < 0 and 'frame_g2g_delay' in self.boundary():
            self.frame_g2g_delay[0] = self.boundary()['frame_g2g_delay'][1]
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
        self.packet_loss_rate[0] = context.packet_loss_rate()
        self.packet_rtt_mea[0] = context.packet_rtt_measured()
        self.packet_delay[0] = context.packet_delay()
        self.pacing_burst_interval[0] = context.packet_delay_interval()
        self.pacing_rate[0] = context.networking.pacing_rate_data[-1][1] \
                if context.networking.pacing_rate_data else 0
        self.pacing_burst_interval[0] = context.networking.pacing_burst_interval_data[-1][1] \
                if context.networking.pacing_burst_interval_data else 0
        self.codec_bitrate[0] = context.bitrate_data[-1][1] if context.bitrate_data else 0
        self.codec_fps[0] = context.fps_data[-1][1] if context.fps_data else 0
        self.action_gap[0] = action.bitrate - self.packet_egress_rate[0]

    def array(self) -> np.ndarray:
        boundary = Observation.boundary()
        keys = sorted(boundary.keys())
        return np.concatenate([nml(k, getattr(self, k), boundary[k], log=False) for k in keys])

    @staticmethod
    def from_array(array) -> 'Observation':
        observation = Observation()
        boundary = Observation.boundary()
        keys = sorted(boundary.keys())
        for i, k in enumerate(keys):
            setattr(observation, k,
                    dnml(k, array[i * DEFAULT_HISTORY_SIZE:(i + 1) * DEFAULT_HISTORY_SIZE],
                                boundary[k], log=False))
        return observation

    @staticmethod
    def boundary() -> dict:
        # observation used in OnRL
        if use_OnRL:
            return {
                'packet_loss_rate': [0, 1],
                'packet_delay': [0, 1000],
                'packet_delay_interval': [0, 10],
                'packet_ack_rate': [0, 500 * 1024 * 1024],
                'action_gap': [- 500 * 1024 * 1024, 500 * 1024 * 1024],
            }
        return {
            'frame_encoding_delay': [0, 1000],
            'frame_pacing_delay': [0, 1000],
            'frame_decoding_delay': [0, 1000],
            # 'frame_assemble_delay': [0, 1000],
            'frame_g2g_delay': [0, 1000],
            # 'frame_size': [0, 1000_000],
            # 'frame_height': [0, 2160],
            # 'frame_encoded_height': [0, 2160],
            'frame_bitrate': [0, 100_000],
            # 'frame_qp': [0, 255],
            # 'codec': [0, 4],
            'fps': [0, 60],
            'packet_egress_rate': [0, 500 * 1024 * 1024],
            'packet_ack_rate': [0, 500 * 1024 * 1024],
            'packet_loss_rate': [0, 1],
            # 'packet_rtt_mea': [0, 1000],
            'packet_delay': [0, 1000],
            'packet_delay_interval': [0, 10],
            # 'pacing_rate': [0, 500 * 1024 * 1024],
            # 'pacing_burst_interval': [0, 1000],
            # 'codec_bitrate': [0, 10 * 1024],
            # 'codec_fps': [0, 60],
        }

    @staticmethod
    def observation_space(legacy_api=False):
        boundary = Observation.boundary()
        keys = sorted(boundary.keys())
        low = np.ones(len(keys) * DEFAULT_HISTORY_SIZE, dtype=np.float32) \
            * NORMALIZATION_RANGE[0]
        high = np.ones(len(keys) * DEFAULT_HISTORY_SIZE, dtype=np.float32) \
            * NORMALIZATION_RANGE[1]
        if legacy_api:
            return Box(low=low, high=high, dtype=np.float32)
        else:
            return spaces.Box(low=low, high=high, dtype=np.float32)


class Action():
    def __init__(self) -> None:
        self.bitrate = np.array([100, ], dtype=np.int32)
        self.pacing_rate = np.array([500 * 1024, ], dtype=np.int32)
        self.resolution = np.array([720, ], dtype=np.int32)

        self.fps = np.array([30, ], dtype=np.int32)
        self.padding_rate = np.array([0, ], dtype=np.int32)
        self.fec_rate_key = np.array([0, ], dtype=np.int32)
        self.fec_rate_delta = np.array([0, ], dtype=np.int32)

    @staticmethod
    def boundary() -> dict:
        return {
            'bitrate': [10, 2500], # When 2500 is the max bitrate set by WebRTC for 720p video
            # 'fps': [1, 60],
            'pacing_rate': [10, 800 * 1024],
            # 'padding_rate': [0, 500 * 1024],
            # 'fec_rate_key': [0, 255],
            # 'fec_rate_delta': [0, 255],
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


    def write(self, shm) -> None:
        def write_int(value, offset):
            if isinstance(value, np.ndarray):
                value = value[0]
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
        if legacy_api:
            return Box(low=low, high=high, dtype=np.float32)
        else:
            return spaces.Box(low=low, high=high, dtype=np.float32)

    @staticmethod
    def shm_size():
        return 10 * 4


class WebRTCEnv(Env):
    def __init__(self, config={}) -> None:
        self.port = int(config.get('port', 7001))
        assert self.port >= 7001 and self.port <= 7099
        self.sender_log = config.get('sender_log', None)
        self.enable_shm = bool(config.get('enable_shm', True))
        self.legacy_api = bool(config.get('legacy_api', True))
        self.duration = int(config.get('duration', 60))
        self.width = int(config.get('width', 720))
        log(f'WebRTCEnv init with config: {config}')
        self.init_timeout = 8
        self.frame_history_size = 10
        self.packet_history_size = 10
        self.packet_history_duration = 10
        self.step_duration = 1
        self.start_ts = time.time()
        self.step_count = 0
        self.monitor_thread: Thread = None
        self.stop_event: Event = None
        self.context: StreamingContext = None
        self.observation: Observation = None
        self.observation_space = Observation.observation_space(self.legacy_api)
        self.action_space = Action.action_space(self.legacy_api)
        self.process_sender = None
        self.process_receiver = None

    def seed(self, s):
        s = int(s)
        assert 1 <= s <= 99
        self.port = 7000 + s

    def get_observation(self):
        return self.observation.array()

    def shm_name(self):
        if self.enable_shm:
            return f'pandia_{self.port}'
        else:
            return f'pandia_disabled'

    def init_webrtc(self):
        try:
            self.shm = shared_memory.SharedMemory(
                name=self.shm_name(), create=True, size=Action.shm_size())
            log(f'Shared memory created: {self.shm.name}')
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(
                name=self.shm_name(), create=False, size=Action.shm_size())
            log(f'Shared memory opened: {self.shm.name}')
        self.stop_event = Event()
        process = subprocess.Popen([os.path.join(SCRIPTS_PATH, 'start_webrtc_receiver_remote.sh'),
                          '-p', str(self.port), '-d', str(self.duration + 10)], shell=False)
        process.wait()

    def start_webrtc(self):
        self.process_sender = subprocess.Popen([os.path.join(BIN_PATH, 'peerconnection_client_headless'),
                                                '--server', '195.148.127.230',
                                                '--port', str(self.port), '--name', 'sender',
                                                '--width', str(self.width), '--fps', str(30), '--autocall', 'true',
                                                '--force_fieldtrials=WebRTC-FlexFEC-03-Advertised/Enabled/WebRTC-FlexFEC-03/Enabled/'],
                                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        stdout = self.process_sender.stderr
        self.monitor_thread = Thread(
            target=monitor_webrtc_sender, args=(self.context, stdout, self.stop_event, self.sender_log))
        self.monitor_thread.start()

    def stop_wevrtc(self):
        process = subprocess.Popen([os.path.join(SCRIPTS_PATH, 'stop_webrtc_receiver_remote.sh'), '-p', str(self.port)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False)
        process.wait()
        if self.process_sender:
            self.process_sender.kill()
        if self.stop_event and not self.stop_event.isSet():
            self.stop_event.set()

    def reset(self, *, seed=None, options=None):
        self.stop_wevrtc()
        self.step_count = 0
        self.context = StreamingContext()
        self.observation = Observation()
        self.init_webrtc()
        obs = self.get_observation()
        # Wait a bit so that the previous process is killed
        time.sleep(1)
        log(f'#0, Obs.: {self.observation}')
        if self.legacy_api:
            return obs
        else:
            return obs, {}

    def get_action(self):
        action = Action()
        for k in Action.boundary():
            if k == 'bitrate':
                action.bitrate[0] = self.context.action_context.bitrate
            if k == 'pacing_rate':
                action.pacing_rate[0] = self.context.action_context.pacing_rate
            if k == 'resolution':
                action.resolution[0] = self.context.action_context.resolution
        return action

    def step(self, action):
        self.context.reset_action_context()
        action = Action.from_array(action)
        action.write(self.shm)
        if not self.context.codec_initiated:
            assert self.step_count == 0
            self.start_webrtc()
            log('Waiting for WebRTC to be ready...')
            ts = time.time()
            while not self.context.codec_initiated and \
                time.time() - ts < self.init_timeout:
                time.sleep(.1)
            if time.time() - ts >= self.init_timeout:
                log(f'Warning: WebRTC init timeout.')
                if self.legacy_api:
                    return self.get_observation(), 0, True, {}
                else:
                    return self.get_observation(), 0, True, True, {}
            # log('WebRTC is running.')
            self.start_ts = time.time()
        end_ts = self.start_ts + self.step_duration * self.step_count
        sleep_duration = end_ts - time.time()
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        self.observation.append(self.context, action)
        done = self.process_sender.poll() is not None
        if time.time() - self.start_ts > self.duration:
            done = True
        self.step_count += 1
        reward = self.reward(action)
        log(f'#{self.step_count} R.w.: {reward:.02f}, Act.: {action}, Obs.: {self.observation}')
        if self.legacy_api:
            return [self.get_observation(), reward, done, {}]
        else:
            return [self.get_observation(), reward, False, done, {}]

    def detect_safe_condition(self):
        pass

    def close(self):
        self.stop_wevrtc()
        self.shm.close()
        self.shm.unlink()

    def reward(self, action: Action):
        if use_OnRL:
            q = self.observation.packet_ack_rate[0] / 1024
            l = self.observation.packet_loss_rate[0]
            d = self.observation.packet_delay[0] / 10
            p = (self.action_pre.bitrate[0] - action.bitrate[0]) / 1000 if self.action_pre else 0
            self.action_pre = action
            return q - l - d - p
        if self.observation.fps[0] == 0:
            return -10
        delay_score = self.observation.frame_g2g_delay[0] / 100
        quality_score = self.observation.frame_bitrate[0] / 1000
        return quality_score - delay_score


gymnasium.register(
    id='WebRTCEnv-v0',
    entry_point=WebRTCEnv,
    max_episode_steps=60,
)
gym.register(
    id='WebRTCEnv-v0',
    entry_point=WebRTCEnv,
    max_episode_steps=60,
)
