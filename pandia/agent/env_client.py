from io import IOBase
from multiprocessing import Pool, Process, shared_memory
import os
import subprocess
import time
import gymnasium as gym
from typing import List, Optional, TextIO

import numpy as np
from pandia import BIN_PATH, SCRIPTS_PATH
from pandia.agent.env import Action, Observation
from ray.rllib.env.policy_client import PolicyClient
from pandia.agent.env_server import SERVER_PORT
from pandia.log_analyzer_sender import StreamingContext, parse_line
from pandia.ntp.ntpclient import ntp_sync


class ActionHistory():
    def __init__(self) -> None:
        pass

    def append(self, action: Action) -> None:
        pass


class WebRTCEnv0(gym.Env):
    def __init__(self, client_id=1, duration=30, # Exp settings
                 width=1080, fps=30, # Source settings
                 bw=1024 * 1024, delay=5, loss=0, # Network settings
                 working_dir=None, # Logging settings
                 step_duration=1, # RL settings
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
        # RL settings
        self.step_duration = step_duration
        self.init_timeout = 5
        self.hisory_size = 10
        # RL state
        self.step_count = 0
        self.shm: shared_memory.SharedMemory
        self.context: StreamingContext
        self.observation: Observation
        self.process_sender: Optional[subprocess.Popen] = None
        # ENV state
        self.action_space = Action.action_space(legacy_api=False) # type: ignore
        self.observation_space = Observation.observation_space(legacy_api=False) # type: ignore
        # Tracking
        self.last_ts = time.time()
        self.stdout: TextIO 
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
        time.sleep(1)

    def start_webrtc(self):
        self.process_traffic_control = \
            subprocess.Popen([os.path.join(SCRIPTS_PATH, 'start_traffic_control_remote.sh'),
                                '-p', str(self.port), '-b', str(self.bw), '-d', str(self.delay),])
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

    def reward(self):
        if self.observation.fps[0] == 0:
            return -10
        delay_score = self.observation.frame_g2g_delay[0] / 100
        quality_score = self.observation.frame_bitrate[0] / 1000
        resolution_penalty = 0
        # # Donot penalize if the resolution is changed less than 1 time during the last 10 steps
        # if resolution_penalty <= 1:
        #     resolution_penalty = 0

        return quality_score - delay_score - resolution_penalty

    def reset(self, seed=None, options=None):
        self.stop_webrtc()
        self.action_history = ActionHistory()
        self.step_count = 0
        self.context = StreamingContext()
        self.context.utc_offset = ntp_sync().offset
        self.observation = Observation()
        self.init_webrtc()
        return self.observation.array(), {}


    def step(self, action: np.ndarray):
        # Write action
        self.context.reset_action_context()
        act = Action.from_array(action)
        self.action_history.append(act)
        act.write(self.shm)

        # Start WebRTC at the first step
        if self.step_count == 0:
            self.stdout = self.start_webrtc()
            self.last_ts = time.time()

        # Process logs and wait for step duration.
        # The first step may last longer than 
        # step duration until the codec is initiated.
        while not self.context.codec_initiated or \
            time.time() - self.last_ts < self.step_duration:
            if time.time() - self.last_ts > self.init_timeout:
                print(f"[{self.client_id}] WebRTC start timeout. Startover.")
                self.reset()
                return self.step(action)
            line = self.stdout.readline().decode().strip()
            if line:
                if self.sender_log:
                    with open(self.sender_log, 'a+') as f:
                        f.write(line + '\n')
                codec_initiated = self.context.codec_initiated
                parse_line(line, self.context)
                if not codec_initiated and self.context.codec_initiated:
                    self.last_ts = time.time()
        self.last_ts = time.time()

        # Collection rollouts
        self.observation.append(self.context, act)
        truncated = self.process_sender.poll() is not None
        reward = self.reward()

        print(f'[{self.client_id}] #{self.step_count} R.w.: {reward:.02f}, Act.: {act}, Obs.: {self.observation}')
        self.step_count += 1
        return self.observation.array(), reward, False, truncated, {}

    def close(self):
        self.stop_webrtc()
        if self.shm:
            self.shm.close()
            self.shm.unlink()


def run(client_id=1):
    random_action=False
    client_id = int(client_id)
    client = PolicyClient(
        f"http://localhost:{SERVER_PORT+client_id}", inference_mode='remote'
    )
    env = WebRTCEnv0(client_id=client_id, duration=60)
    obs, info = env.reset()
    action_space = Action.action_space(legacy_api=False)
    eid = client.start_episode()
    rewards = 0
    while True:
        if random_action:
            action = action_space.sample()
            client.log_action(eid, obs, action)
        else:
            action = client.get_action(eid, obs)
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
