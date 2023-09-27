from multiprocessing import shared_memory
import os
import socket
import subprocess
import time
from typing import Optional
import uuid
import gymnasium
import numpy as np

from ray import tune
from pandia import BIN_PATH
from pandia.agent.action import Action
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.env_container import ObservationThread
from pandia.agent.observation import Observation
from pandia.agent.reward import reward
from pandia.log_analyzer_sender import StreamingContext


class WebRTCSimulatorEnv(gymnasium.Env):
    def __init__(self, client_id=None, rank=None, duration=ENV_CONFIG['duration'], # Exp settings
                 resolution=ENV_CONFIG['width'], fps=ENV_CONFIG['fps'], # Source settings
                 bw=ENV_CONFIG['bandwidth_range'],  # Network settings
                 delay=ENV_CONFIG['delay_range'], loss=ENV_CONFIG['loss_range'], # Network settings
                 action_keys=ENV_CONFIG['action_keys'], # Action settings
                 obs_keys=ENV_CONFIG['observation_keys'], # Observation settings
                 monitor_durations=ENV_CONFIG['observation_durations'], # Observation settings
                 print_step=True, print_period=2, log_path=None,# Logging settings
                 step_duration=ENV_CONFIG['step_duration'], # RL settings
                 termination_timeout=ENV_CONFIG['termination_timeout'] # Exp settings
                 ) -> None:
        super().__init__()
        # Exp settings
        self.uuid = str(uuid.uuid4())[:8]
        self.duration = duration
        self.termination_timeout = termination_timeout
        # Source settings
        self.resolution = resolution
        self.fps = fps
        # Network settings
        self.bw0 = bw  # in kbps
        self.delay0 = delay  # in ms
        self.loss0 = loss  # in %
        self.net_params = {}
        # Logging settings
        self.print_step = print_step
        self.print_period = print_period
        self.log_path = log_path
        # RL settings
        self.step_duration = step_duration
        self.init_timeout = 10
        self.hisory_size = 1
        # RL state
        self.step_count = 0
        try:
            self.shm = shared_memory.SharedMemory(name=self.shm_name, 
                                                  create=True, size=Action.shm_size())
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=self.shm_name, 
                                                  create=False, size=Action.shm_size())
        self.context: StreamingContext
        self.obs_keys = list(sorted(obs_keys))
        self.monitor_durations = list(sorted(monitor_durations))
        self.observation: Observation = Observation(self.obs_keys, 
                                                    durations=self.monitor_durations,
                                                    history_size=self.hisory_size)
        # ENV state
        self.termination_ts = 0
        self.action_keys = list(sorted(action_keys))
        self.action_space = Action(action_keys).action_space()
        self.observation_space = \
            Observation(self.obs_keys, self.monitor_durations, self.hisory_size)\
                .observation_space()
        # Tracking
        self.process: Optional[subprocess.Popen] = None
        self.obs_socket = self.create_observer()
        self.obs_thread = ObservationThread(self.obs_socket)
        self.obs_thread.start()

    @property
    def shm_name(self):
        return f"pandia_{self.uuid}"

    @property
    def obs_socket_path(self):
        return f'/tmp/{self.uuid}.sock'

    def log(self, msg):
        print(f'[{self.uuid}, {time.time() - self.start_ts:.02f}] {msg}', flush=True)

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

    def start_webrtc(self):
        if self.log_path:
            output = open(self.log_path, 'w')
        else:
            output = subprocess.DEVNULL
        self.process = \
            subprocess.Popen([os.path.join(BIN_PATH, 'pandia'), 
                              '--shm_name', self.shm_name,
                              '--obs_socket', self.obs_socket_path,
                              '--resolution', str(self.resolution), 
                              '--fps', str(self.fps),
                              '--duration', str(self.duration),
                              '--logging_path', '/tmp/pandia.log',
                              '--force_fieldtrials=WebRTC-FlexFEC-03-Advertised/Enabled/WebRTC-FlexFEC-03/Enabled/', 
                            ], 
                            stdout=output, stderr=output, shell=False)
        
    def create_observer(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.bind(self.obs_socket_path)
        print(f'Listening on IPC socket {self.obs_socket_path}')
        return sock

    def reset(self, seed=None, options=None):
        if self.process is not None:
            self.process.kill()
            self.process = None
        self.context = StreamingContext(monitor_durations=self.monitor_durations)
        self.obs_thread.context = self.context
        self.termination_ts = 0
        self.step_count = 0
        self.last_print_ts = 0
        self.observation = Observation(self.obs_keys, self.monitor_durations, self.hisory_size)
        self.start_ts = time.time()
        return self.observation.array(), {}

    def close(self):
        if os.path.exists(self.obs_socket_path):
            os.remove(self.obs_socket_path)
        if self.process is not None:
            self.process.kill()
        self.obs_thread.stop()

    def step(self, action: np.ndarray):
        self.context.reset_step_context()
        act = Action.from_array(action, self.action_keys)
        act.write(self.shm)
        if self.step_count == 0:
            self.start_webrtc()
        ts = time.time()
        while not self.context.codec_initiated:
            if time.time() - ts > self.init_timeout:
                print(f"WebRTC start timeout. Startover.", flush=True)
                self.reset()
                return self.step(action)
        ts = time.time()
        if self.step_count == 0:
            self.start_ts = ts
            print(f'WebRTC is running.', flush=True)
        time.sleep(self.step_duration)

        self.observation.append(self.context.monitor_blocks, act)
        truncated = time.time() - self.start_ts > self.duration
        r = reward(self.context)

        if self.print_step:
            if time.time() - self.last_print_ts > self.print_period:
                self.last_print_ts = time.time()
                self.log(f'#{self.step_count}@{int((time.time() - self.start_ts))}s '
                      f'R.w.: {r:.02f}, Act.: {act}Obs.: {self.observation}')
        self.step_count += 1
        terminated = self.termination_ts > 0 and self.context.last_ts > self.termination_ts
        return self.observation.array(), r, terminated, truncated, {}


tune.register_env('WebRTCSimulatorEnv', lambda config: WebRTCSimulatorEnv(**config))
gymnasium.register('WebRTCSimulatorEnv', entry_point='pandia.agent.env_simulator:WebRTCSimulatorEnv', nondeterministic=True)


def test():
    env = gymnasium.make("WebRTCSimulatorEnv", 
                         bw=1024 * 1000, delay=0,
                         log_path='/tmp/pandia.log')
    action = Action(ENV_CONFIG['action_keys'])
    action.bitrate = 1024
    action.pacing_rate = 2048000
    episodes = 1
    try:
        for _ in range(episodes):
            env.reset()
            while True:
                _, _, terminated, truncated, _ = env.step(action.array())
                if terminated or truncated:
                    break
    except KeyboardInterrupt:
        pass
    env.close()


if __name__ == '__main__':
    test()
