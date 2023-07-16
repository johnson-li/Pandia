import os
import subprocess
import threading
from threading import Event, Thread
import time
from typing import List, Optional
import numpy as np
import gymnasium
from gymnasium import Env, spaces
from multiprocessing import shared_memory, current_process
from pandia import RESULTS_PATH, SCRIPTS_PATH, BIN_PATH
from pandia.log_analyzer_sender import CODEC_NAMES, ActionContext, FrameContext, PacketContext, StreamingContext, parse_line
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
        self.step_duration = .01
        self.start_ts = time.time()
        self.step_count = 0
        self.monitor_thread: Thread
        self.stop_event: Optional[Event] = None
        self.context: StreamingContext
        self.observation: Observation
        self.observation_space = Observation.observation_space(self.legacy_api)
        self.action_space = Action.action_space(self.legacy_api)
        self.process_sender: subprocess.Popen

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
        print(delay_score)
        penalty = 0
        if delay_score < 0:
            penalty = 10
        quality_score = self.observation.frame_bitrate[0] / 1000
        return quality_score - delay_score - penalty


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
