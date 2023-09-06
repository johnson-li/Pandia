import logging
from multiprocessing import shared_memory
import os
import socket
import subprocess
import threading
import time
from typing import Optional
import gymnasium
import numpy as np
from requests import request
import requests
from pandia.agent.action import Action
from pandia.agent.env_client import ActionHistory, ReadingThread
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.env_server import SERVER_PORT
from pandia.agent.observation import Observation
from pandia.agent.reward import reward
from pandia.agent.utils import sample
from pandia.log_analyzer_sender import StreamingContext 
from ray.rllib.env.policy_client import PolicyClient

logging.getLogger('ray.rllib.env.policy_client').setLevel(logging.WARNING)


def log(msg):
    print(msg, flush=True)


class WebRTCEnv(gymnasium.Env):
    def __init__(self, receiver_ip='127.0.0.1', duration=ENV_CONFIG['duration'], # Exp settings
                 height=ENV_CONFIG['width'], fps=ENV_CONFIG['fps'], # Video source settings
                 bw=ENV_CONFIG['bandwidth_range'],  # Network settings
                 delay=ENV_CONFIG['delay_range'], loss=ENV_CONFIG['loss_range'], # Network settings
                 action_keys=ENV_CONFIG['action_keys'], # Action settings
                 obs_keys=ENV_CONFIG['observation_keys'], # Observation settings
                 monitor_durations=ENV_CONFIG['observation_durations'], # Observation settings
                 print_step=False, print_sender_log=False, print_period=1, # Logging settings
                 sender_log=None, # Logging settings
                 step_duration=ENV_CONFIG['step_duration'], # RL settings
                 termination_timeout=ENV_CONFIG['termination_timeout'] # Exp settings
                 ) -> None:
        super().__init__()
        # Exp settings
        self.receiver_ip = receiver_ip
        self.duration = duration
        self.termination_timeout = termination_timeout
        # Source settings
        self.height = height
        self.fps = fps
        # Network settings
        self.bw0 = bw  # in kbps
        self.delay0 = delay  # in ms
        self.loss0 = loss  # in %
        self.net_params = {}
        # Logging settings
        self.print_step = print_step
        self.print_sender_log = print_sender_log
        self.print_period = print_period
        self.stop_event: Optional[threading.Event] = None
        self.reading_thread: ReadingThread
        self.logging_buf = []
        self.last_print_ts = 0  
        self.sender_log = sender_log
        # RL settings
        self.step_duration = step_duration
        self.init_timeout = 3
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
        # Init shared memory
        self.shm = shared_memory.SharedMemory(name="pandia", create=True, 
                                              size=Action.shm_size())

    def sample_net_params(self):
        return {'bw': sample(self.bw0), 
                'delay': sample(self.delay0), 
                'loss': sample(self.loss0)
                }

    def restart_receiver(self):
        r = requests.post(f'http://{self.receiver_ip}:9998/reset', json={'latency': self.net_params['delay']})
        log(f'Reset received: {r.text}, wait for 1s...')
        time.sleep(1)

    def reset(self, seed=None, options=None):
        if self.process_sender and self.process_sender.poll() is None:
            self.process_sender.kill()
        if self.stop_event:
            self.stop_event.set()
            self.reading_thread.join()
        self.net_params = self.sample_net_params()
        log(f'Net params: {self.net_params}')
        self.restart_receiver()
        self.context = StreamingContext(monitor_durations=self.monitor_durations)
        self.stop_event = threading.Event()
        self.reading_thread = ReadingThread(self.context, self.sender_log, self.stop_event, self.print_sender_log)
        self.reading_thread.start()
        self.action_history = ActionHistory()
        self.termination_ts = 0
        self.step_count = 0
        self.observation = Observation(self.obs_keys, self.monitor_durations, self.hisory_size)
        self.start_ts = time.time()
        return self.observation.array(), {}

    def close(self):
        if self.shm:
            self.shm.close()
            self.shm.unlink()

    def start_webrtc(self):
        log("Starting WebRTC...")
        bw = self.net_params['bw']
        os.system(f"tc qdisc del dev eth0 root")
        os.system(f"tc qdisc add dev eth0 root tbf rate {bw}kbit burst 1000kb minburst 1540 latency 250ms")
        self.process_sender = \
            subprocess.Popen(['/app/peerconnection_client_headless',
                              '--server', self.receiver_ip,
                              '--width', str(self.height), '--fps', str(self.fps),
                              '--force_fieldtrials=WebRTC-FlexFEC-03-Advertised/Enabled/WebRTC-FlexFEC-03/Enabled/', 
                              '--path', '/app/media'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        return self.process_sender.stdout

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

        ts = time.time()
        while not self.context.codec_initiated:
            if time.time() - ts > self.init_timeout:
                log(f"WebRTC start timeout. Startover.")
                self.reset()
                return self.step(action)
        ts = time.time()
        if self.step_count == 0:
            self.start_ts = ts
            log(f'WebRTC is running.')
        time.sleep(self.step_duration)

        self.observation.append(self.context.monitor_blocks, act)
        truncated = self.process_sender.poll() is not None or \
            time.time() - self.start_ts > self.duration
        r = reward(self.context)
        if self.print_step:
            if time.time() - self.last_print_ts > self.print_period:
                self.last_print_ts = time.time()
                log(f'#{self.step_count}@{int((time.time() - self.start_ts))}s '
                    f'R.w.: {r:.02f}, Act.: {act}Obs.: {self.observation}')
        self.step_count += 1
        terminated = self.termination_ts > 0 and self.context.last_ts > self.termination_ts
        return self.observation.array(), r, terminated, truncated, {}



def parse_rangable_int(value):
    if type(value) is str and '-' in value:
        return [int(v) for v in value.split('-')]
    else:
        return int(value)

def main():
    log('Starting RLlib client...')
    rl_server = os.getenv('RL_SERVER', '127.0.0.1')
    print_step = bool(os.getenv('PRINT_STEP', False))
    sender_log = os.getenv('SENDER_LOG', None)
    bw = parse_rangable_int(os.getenv('BANDWIDTH', 3000))
    delay = parse_rangable_int(os.getenv('DELAY', 0))
    loss = parse_rangable_int(os.getenv('LOSS', 0))
    random_action= bool(os.getenv('RANDOM_ACTION', False))
    print_sender_log = bool(os.getenv('PRINT_SENDER_LOG', False))

    hostname = socket.gethostname()
    receiver_name = hostname.replace('sender', 'receiver')
    log(f'Receiver name: {receiver_name}')
    receiver_ip = socket.gethostbyname(receiver_name) 
    client = PolicyClient(
        f"http://{rl_server}:{SERVER_PORT}", inference_mode='local'
    )
    if sender_log:
        sender_log = f'{sender_log}_{hostname}'
    env = WebRTCEnv(receiver_ip=receiver_ip, print_step=print_step, 
                    bw=bw, delay=delay, loss=loss, sender_log=sender_log,
                    print_sender_log=print_sender_log)
    obs, info = env.reset()
    eid = client.start_episode()
    rewards = 0
    while True:
        if random_action:
            action_space = Action(ENV_CONFIG['action_keys']).action_space()
            action: np.ndarray = action_space.sample()
            client.log_action(eid, obs, action)
        else:
            action: np.ndarray = client.get_action(eid, obs) # type: ignore
        obs, reward, terminated, truncated, info = env.step(action)
        rewards += reward
        client.log_returns(eid, reward, info=info)
        if terminated or truncated:
            log(f'Total reward after {env.step_count} steps: {rewards:.02f}')
            rewards = 0
            client.end_episode(eid, obs)
            obs, info = env.reset()
            eid = client.start_episode()


if __name__ == "__main__":
    main()
