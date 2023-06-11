from multiprocessing import Pool, Process, shared_memory
import os
import subprocess
import time
from typing import TextIO
from pandia import BIN_PATH, SCRIPTS_PATH
from pandia.agent.env import Action, Observation
from ray.rllib.env.policy_client import PolicyClient
from pandia.agent.env_server import SERVER_PORT
from pandia.log_analyzer import StreamingContext, parse_line


class Env():
    def __init__(self, client_id=1, duration=30, width=720,
                 step_duration=1,) -> None:
        self.client_id = client_id
        self.port = 7000 + client_id
        self.duration = duration
        self.width = width
        self.init_timeout = 5
        self.hisory_size = 10
        self.step_duration = step_duration
        self.last_ts = time.time()
        self.step_count = 0
        self.context: StreamingContext
        self.observation: Observation
        self.process_sender: subprocess.Popen = None
        self.stdout: TextIO 

    @property
    def shm_name(self):
        return f"pandia_{self.port}"

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
        process = subprocess.Popen([os.path.join(SCRIPTS_PATH, 'start_webrtc_receiver_remote.sh'), 
                          '-p', str(self.port), '-d', str(self.duration + 10)], shell=False)
        process.wait()
        time.sleep(1)

    def start_webrtc(self):
        self.process_sender = subprocess.Popen([os.path.join(BIN_PATH, 'peerconnection_client_headless'),
                                                '--server', '195.148.127.230',
                                                '--port', str(self.port), '--name', 'sender',
                                                '--width', str(self.width), '--fps', str(30), '--autocall', 'true',
                                                '--force_fieldtrials=WebRTC-FlexFEC-03-Advertised/Enabled/WebRTC-FlexFEC-03/Enabled/'],
                                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        return self.process_sender.stderr

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
        return quality_score - delay_score

    def reset(self):
        self.stop_webrtc()
        self.step_count = 0
        self.context = StreamingContext()
        self.observation = Observation()
        self.init_webrtc()
        return self.observation.array(), {}


    def step(self, action):
        # Write action
        self.context.reset_action_context()
        act = Action.from_array(action)
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
                codec_initiated = self.context.codec_initiated
                parse_line(line, self.context)
                if not codec_initiated and self.context.codec_initiated:
                    self.last_ts = time.time()
        self.last_ts = time.time()

        # Collection rollouts
        self.observation.append(self.context)
        truncated = self.process_sender.poll() is not None
        reward = self.reward()

        print(f'[{self.client_id}] #{self.step_count} R.w.: {reward:.02f}, Act.: {act}, Obs.: {self.observation}')
        self.step_count += 1
        return self.observation.array(), reward, False, truncated, {}

    def close(self):
        self.stop_webrtc()
        self.shm.close()
        self.shm.unlink()


def run(client_id=1):
    random_action=False
    client_id = int(client_id)
    client = PolicyClient(
        f"http://localhost:{SERVER_PORT+client_id}", inference_mode='remote'
    )
    env = Env(client_id=client_id, duration=60)
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
            print(f'[{client_id}] Total reward after {env.step_count} steps: {rewards}')
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
