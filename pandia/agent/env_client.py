from io import IOBase
from multiprocessing import Pool, Process, shared_memory
import os
import subprocess
from threading import Thread
import threading
import time
import gymnasium as gym
from typing import List, Optional, TextIO, Union

import numpy as np
from pandia import BIN_PATH, SCRIPTS_PATH
from pandia.agent.action import Action
from pandia.agent.observation import Observation
from ray.rllib.env.policy_client import PolicyClient
from pandia.agent.env_server import SERVER_PORT
from pandia.log_analyzer_sender import StreamingContext, parse_line
from pandia.ntp.ntpclient import NTP_OFFSET_PATH, ntp_sync


class ActionHistory():
    def __init__(self) -> None:
        pass

    def append(self, action: Action) -> None:
        pass


class ReadingThread(threading.Thread):
    def __init__(self, context: StreamingContext, log_file: Optional[str], stop_event: threading.Event):
        super().__init__()
        self.stdout = None
        self.context = context
        self.log_file = log_file
        self.stop_event = stop_event


    def run(self) -> None:
        buf = []
        while not self.stop_event.is_set():
            if self.stdout:
                line = self.stdout.readline().decode().strip()
                if line:
                    parse_line(line, self.context)
                    if self.log_file:
                        buf.append(line)
        if self.log_file:
            with open(self.log_file, 'w+') as f:
                f.write('\n'.join(buf))


class WebRTCEnv0(gym.Env):
    def __init__(self, client_id=1, duration=30, # Exp settings
                 width=2160, fps=60, # Source settings
                 bw=1024 * 1024, delay=5, loss=2, # Network settings
                 action_keys=Action.boundary().keys(), # Action settings
                 obs_keys=Observation.boundary().keys(), # Observation settings
                 monitor_durations=[1, 2, 4], # Observation settings
                 working_dir=None, # Logging settings
                 step_duration=.01, # RL settings
                 ) -> None:
        super().__init__() 
        # Exp settings
        self.client_id = client_id
        self.port = 7000 + client_id
        self.duration = duration
        # Source settings
        self.width = width
        self.fps = fps
        # Network settings
        self.bw = bw
        self.delay = delay
        self.loss = loss
        # Logging settings
        self.working_dir = working_dir
        self.sender_log = os.path.join(working_dir, self.log_name('sender')) if working_dir else None
        self.stop_event: Optional[threading.Event] = None
        self.reading_thread: ReadingThread
        self.logging_buf = []
        # RL settings
        self.step_duration = step_duration
        self.init_timeout = 10
        self.hisory_size = 1
        # RL state
        self.step_count = 0
        self.shm: shared_memory.SharedMemory
        self.context: StreamingContext
        self.obs_keys = list(sorted(obs_keys))
        self.monitor_durations = list(sorted(monitor_durations))
        self.observation: Observation = Observation(self.obs_keys, durations=self.monitor_durations, 
                                                    history_size=self.hisory_size)
        self.process_sender: Optional[subprocess.Popen] = None
        # ENV state
        self.action_keys = list(sorted(action_keys))
        self.action_space = Action(action_keys).action_space()
        self.observation_space = \
            Observation(self.obs_keys, self.monitor_durations, self.hisory_size)\
                .observation_space()
        # Tracking
        self.start_ts = 0
        self.action_history = ActionHistory()

    @property
    def shm_name(self):
        return f"pandia_{self.port}"
    
    def log_name(self, role='sender'):
        return f"eval_{role}_{self.port}.log"

    def init_webrtc(self):
        shm_size = Action.shm_size()
        print(f"[{self.client_id}] Initializing shm {self.shm_name} for WebRTC")
        try:
            self.shm = \
                shared_memory.SharedMemory(name=self.shm_name, 
                                        create=True, size=shm_size)
        except FileExistsError:
            self.shm = \
                shared_memory.SharedMemory(name=self.shm_name, 
                                        create=False, size=shm_size)
        receiver_log = f'/tmp/{self.log_name("receiver")}' if self.working_dir else '/dev/null'
        process = subprocess.Popen([os.path.join(SCRIPTS_PATH, 'start_webrtc_receiver_remote.sh'), 
                          '-p', str(self.port), '-d', str(self.duration + 3), 
                          '-l', receiver_log], shell=False)
        process.wait()
        # time.sleep(1)

    def start_webrtc(self):
        self.process_traffic_control = \
            subprocess.Popen([os.path.join(SCRIPTS_PATH, 'start_traffic_control_remote.sh'),
                                '-p', str(self.port), '-b', str(self.bw), '-d', str(self.delay), '-l', str(self.loss)])
        self.process_traffic_control.wait()
        self.process_sender = subprocess.Popen([os.path.join(BIN_PATH, 'peerconnection_client_headless'),
                                                '--server', '195.148.127.230',
                                                '--port', str(self.port), '--name', 'sender',
                                                '--width', str(self.width), '--fps', str(self.fps), '--autocall', 'true',
                                                '--force_fieldtrials=WebRTC-FlexFEC-03-Advertised/Enabled/WebRTC-FlexFEC-03/Enabled/'],
                                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        return self.process_sender.stdout

    def stop_webrtc(self):
        process = subprocess.Popen([os.path.join(SCRIPTS_PATH, 'stop_webrtc_receiver_remote.sh'), '-p', str(self.port)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False)
        process.wait()
        time.sleep(1)
        if self.process_sender:
            self.process_sender.kill()
        if self.sender_log and self.logging_buf:
            with open(self.sender_log, 'w+') as f:
                f.write('\n'.join(self.logging_buf))
        if self.stop_event is not None:
            self.stop_event.set()
            self.reading_thread.join()

    def reward(self):
        mb = self.context.monitor_blocks[self.monitor_durations[0]]
        if mb.frame_fps == 0:
            return -10
        penalty = 0 
        delay_score = mb.frame_decoded_delay * 10
        if delay_score < 0:
            penalty = 10
        quality_score = mb.frame_bitrate / 1024 / 1024
        return quality_score - delay_score - penalty

    def reset(self, seed=None, options=None):
        self.stop_webrtc()
        if self.sender_log and os.path.exists(self.sender_log):
            os.remove(self.sender_log)
        self.context = StreamingContext(monitor_durations=self.monitor_durations)
        self.stop_event = threading.Event()
        self.reading_thread = ReadingThread(self.context, self.sender_log, self.stop_event)
        self.reading_thread.start()
        self.action_history = ActionHistory()
        self.step_count = 0
        if os.path.isfile(NTP_OFFSET_PATH):
            data = open(NTP_OFFSET_PATH, 'r').read().split(',')
            self.context.update_utc_offset(float(data[0]))
        self.observation = Observation(self.obs_keys, self.monitor_durations, self.hisory_size)
        self.init_webrtc()
        self.start_ts = time.time()
        return self.observation.array(), {}

    def close(self):
        self.stop_webrtc()
        if self.shm:
            self.shm.close()
            self.shm.unlink()

    def step(self, action: np.ndarray):
        # Write action
        self.context.reset_step_context()
        act = Action.from_array(action, self.action_keys)
        self.action_history.append(act)
        act.write(self.shm)

        # Start WebRTC at the first step
        if self.step_count == 0:
            self.reading_thread.stdout = self.start_webrtc()
            print(f"[{self.client_id}] Wait for WebRTC to start.")

        ts = time.time()
        while not self.context.codec_initiated:
            if time.time() - ts > self.init_timeout:
                print(f"[{self.client_id}] WebRTC start timeout. Startover.")
                self.reset()
                return self.step(action)
        ts = time.time()
        if self.step_count == 0:
            self.start_ts = ts
        time.sleep(self.step_duration)

        self.observation.append(self.context.monitor_blocks, act)
        truncated = self.process_sender.poll() is not None or \
            time.time() - self.start_ts > self.duration
        reward = self.reward()

        print(f'[{self.client_id}] #{self.step_count}@{int((time.time() - self.start_ts))}s '
              f'R.w.: {reward:.02f}, Act.: {act}Obs.: {self.observation}')
        self.step_count += 1
        return self.observation.array(), reward, False, truncated, {}


def run(client_id=1):
    random_action=False
    client_id = int(client_id)
    client = PolicyClient(
        f"http://localhost:{SERVER_PORT+client_id}", inference_mode='remote'
    )
    env = WebRTCEnv0(client_id=client_id, duration=60)
    obs, info = env.reset()
    action_space = Action.action_space()
    eid = client.start_episode()
    rewards = 0
    while True:
        if random_action:
            action: np.ndarray = action_space.sample()
            client.log_action(eid, obs, action)
        else:
            action: np.ndarray = client.get_action(eid, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards += reward
        client.log_returns(eid, reward, info=info)
        if terminated or truncated:
            print(f'[{client_id}] Total reward after {env.step_count} steps: {rewards:.02f}')
            rewards = 0
            client.end_episode(eid, obs)
            obs, info = env.reset()
            eid = client.start_episode()


def main():
    concurrency = 8
    with Pool(concurrency) as p:
        p.map(run, [i + 1 for i in range(concurrency)])


if __name__ == '__main__':
    main()
