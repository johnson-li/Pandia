import argparse
from io import IOBase
from multiprocessing import Pool, Process, shared_memory
from ray import tune
import os
import subprocess
from threading import Thread
import threading
import time
import gymnasium as gym
from typing import List, Optional, TextIO, Union
from pandia.agent.env_config import ENV_CONFIG
from pandia.constants import K, M
import numpy as np
from pandia import BIN_PATH, SCRIPTS_PATH
from pandia.agent.action import Action
from pandia.agent.observation import Observation
from ray.rllib.env.policy_client import PolicyClient
from pandia.agent.env_server import SERVER_PORT
from pandia.log_analyzer_sender import StreamingContext, parse_line


class ActionHistory():
    def __init__(self) -> None:
        pass

    def append(self, action: Action) -> None:
        pass


class ReadingThread(threading.Thread):
    def __init__(self, context: StreamingContext, log_file: Optional[str], 
                 stop_event: threading.Event, log_stdout: bool = False) -> None:
        super().__init__()
        self.stdout = None
        self.context = context
        self.log_file = log_file
        self.stop_event = stop_event
        self.log_stdout = log_stdout


    def run(self) -> None:
        buf = []
        while not self.stop_event.is_set():
            if self.stdout:
                line = self.stdout.readline().decode().strip()
                if line:
                    if self.log_stdout:
                        print(line, flush=True)
                    parse_line(line, self.context)
                    if self.log_file:
                        buf.append(line)
        if self.log_file:
            # print(f'Dump log to {self.log_file}, lines: {len(buf)}')
            with open(self.log_file, 'w+') as f:
                f.write('\n'.join(buf))


class WebRTCEnv0(gym.Env):
    def __init__(self, client_id=None, rank=None, duration=ENV_CONFIG['duration'], # Exp settings
                 width=ENV_CONFIG['width'], fps=ENV_CONFIG['fps'], # Source settings
                 bw=ENV_CONFIG['bandwidth_range'],  # Network settings
                 delay=ENV_CONFIG['delay_range'], loss=ENV_CONFIG['loss_range'], # Network settings
                 action_keys=ENV_CONFIG['action_keys'], # Action settings
                 obs_keys=ENV_CONFIG['observation_keys'], # Observation settings
                 monitor_durations=ENV_CONFIG['observation_durations'], # Observation settings
                 working_dir=None, print_step=False,# Logging settings
                 step_duration=ENV_CONFIG['step_duration'], # RL settings
                 termination_timeout=ENV_CONFIG['termination_timeout'] # Exp settings
                 ) -> None:
        super().__init__()
        print(f'Creating WebRTCEnv0 with client_id={client_id}, rank={rank}')
        # Exp settings
        self.client_id: int = client_id if client_id is not None else 1
        if rank is not None:
            self.client_id = rank + 1
        assert not os.path.exists(self.placeholder_path), f'Placeholder {self.placeholder_path} exists.'
        with open(self.placeholder_path, 'w+') as f:
            f.write('placeholder')
        self.port = 7000 + self.client_id
        self.duration = duration
        self.termination_timeout = termination_timeout
        # Source settings
        self.width = width
        self.fps = fps
        # Network settings
        self.bw0 = bw  # in kbps
        self.delay0 = delay  # in ms
        self.loss0 = loss  # in %
        self.net_params = {}
        # Logging settings
        self.print_step = print_step
        self.working_dir = working_dir
        if self.working_dir and not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
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
        self.termination_ts = 0
        self.action_keys = list(sorted(action_keys))
        self.action_space = Action(action_keys).action_space()
        self.observation_space = \
            Observation(self.obs_keys, self.monitor_durations, self.hisory_size)\
                .observation_space()
        # print(f'Action shape: {self.action_space.shape}, observation shape: {self.observation_space.shape}')
        # Tracking
        self.start_ts = 0
        self.action_history = ActionHistory()
        self.penalty_timeout = 0

    @property
    def placeholder_path(self):
        return f'/tmp/pandia_placeholder_{self.client_id}'

    def sample_net_params(self):
        if type(self.bw0) is list:
            bw = int(np.random.uniform(self.bw0[0], self.bw0[1]))
        else:
            bw = self.bw0
        if type(self.delay0) is list:
            delay = int(np.random.uniform(self.delay0[0], self.delay0[1]))
        else:
            delay = self.delay0
        if type(self.loss0) is list:
            loss = int(np.random.uniform(self.loss0[0], self.loss0[1]))
        else:
            loss = self.loss0
        return {'bw': bw, 'delay': delay, 'loss': loss}

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
        bw = self.net_params['bw']
        delay = self.net_params['delay']
        loss = self.net_params['loss']
        self.process_traffic_control = \
            subprocess.Popen([os.path.join(SCRIPTS_PATH, 'start_traffic_control_remote.sh'),
                                '-p', str(self.port), '-b', str(bw),
                                '-d', str(delay), '-l', str(loss)])
        self.process_traffic_control.wait()
        self.process_sender = \
            subprocess.Popen([os.path.join(BIN_PATH, 'peerconnection_client_headless'),
                              '--server', '195.148.127.230',
                              '--port', str(self.port), '--name', 'sender',
                              '--width', str(self.width), '--fps', str(self.fps), '--autocall', 'true',
                              '--force_fieldtrials=WebRTC-FlexFEC-03-Advertised/Enabled/WebRTC-FlexFEC-03/Enabled/', 
                              '--path', os.path.expanduser('~/Downloads')],
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

    @staticmethod
    def reward(context: StreamingContext, terminated=False):
        monitor_durations = list(sorted(context.monitor_blocks.keys()))
        mb = context.monitor_blocks[monitor_durations[0]]
        penalty = 0
        # fps_score = mb.frame_fps / 30
        fps_score = 0
        delay_score = 0
        for delay in [mb.frame_decoded_delay, mb.frame_egress_delay]:
            delay *= 1000
            delay_score += - delay ** 2 / 100 ** 2
        quality_score = mb.frame_bitrate / 1024 / 1024
        # res_score = mb.frame_height / 2160
        res_score = 0
        # if penalty == 0:
        #     self.termination_ts = 0
        # if penalty > 0 and self.termination_ts == 0:
        #     # If unexpected situation lasts for 5s, terminate
        #     self.termination_ts = self.context.last_ts + self.termination_timeout
        # if mb.frame_fps < 1:
        #     penalty = 100
        score = res_score + quality_score + fps_score + delay_score
        score = max(-10, score)
        return score

    def reset(self, seed=None, options=None):
        self.stop_webrtc()
        if self.sender_log and os.path.exists(self.sender_log):
            os.remove(self.sender_log)
        self.net_params = self.sample_net_params()
        self.context = StreamingContext(monitor_durations=self.monitor_durations)
        self.stop_event = threading.Event()
        self.reading_thread = ReadingThread(self.context, self.sender_log, self.stop_event)
        self.reading_thread.start()
        self.action_history = ActionHistory()
        self.termination_ts = 0
        self.step_count = 0
        from pandia.ntp.ntpclient import NTP_OFFSET_PATH
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
        if os.path.isfile(self.placeholder_path):
            os.remove(self.placeholder_path)

    def is_safe(self) -> bool:
        return True

    def step(self, action: np.ndarray):
        # Write action
        self.context.reset_step_context()
        act = Action.from_array(action, self.action_keys)
        self.action_history.append(act)
        if self.penalty_timeout > 0:
            act = Action(['fake'])
            self.penalty_timeout -= 1
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
            print(f'[{self.client_id}] WebRTC is running.')
        time.sleep(self.step_duration)

        self.observation.append(self.context.monitor_blocks, act)
        truncated = self.process_sender.poll() is not None or \
            time.time() - self.start_ts > self.duration
        reward = self.reward(self.context)
        if not self.is_safe():
            self.penalty_timeout = 1

        if self.print_step:
            print(f'[{self.client_id}] #{self.step_count}@{int((time.time() - self.start_ts))}s '
                f'R.w.: {reward:.02f}, Act.: {act}Obs.: {self.observation}')
        self.step_count += 1
        terminated = self.termination_ts > 0 and self.context.last_ts > self.termination_ts
        return self.observation.array(), reward, terminated, truncated, {}


def run_wrapper(client_id=1, print_step=False):
    try:
        return run(client_id, print_step)
    except Exception as e:
        print(f'[{client_id}] Exception: {e}')
        raise e


def run(client_id=1, print_step=False):
    random_action=False
    client_id = int(client_id)
    client = PolicyClient(
        f"http://localhost:{SERVER_PORT+client_id}", inference_mode='remote'
    )
    env = WebRTCEnv0(client_id=client_id, print_step=print_step, bw=3000)
    obs, info = env.reset()
    eid = client.start_episode()
    rewards = 0
    while True:
        if random_action:
            action_space = Action(ENV_CONFIG['action_keys']).action_space()
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


def test():
    client_id = 1
    rewards = 0
    env = WebRTCEnv0(client_id=client_id, duration=10)
    obs, info = env.reset()
    while True:
        action_space = Action(ENV_CONFIG['action_keys']).action_space()
        action: np.ndarray = action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards += reward
        if terminated or truncated:
            print(f'[{client_id}] Total reward after {env.step_count} steps: {rewards:.02f}')
            break
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--concurrency', type=int, default=8)
    parser.add_argument('-p', '--print_step', action='store_true')
    parser.add_argument('-t', '--test', action='store_true')
    args = parser.parse_args()
    concurrency = args.concurrency
    print_step = args.print_step
    if args.test:
        return test()
    else:
        with Pool(concurrency) as p:
            p.starmap(run_wrapper, [[i + 1, print_step] for i in range(concurrency)])


tune.register_env('pandia', lambda config: WebRTCEnv0(**config))
gym.register('pandia', entry_point='pandia.agent.env_client:WebRTCEnv0', nondeterministic=True)


if __name__ == '__main__':
    main()
