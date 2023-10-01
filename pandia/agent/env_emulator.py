import json
import os
import socket
import docker
import time
import uuid
import gymnasium
import numpy as np

from docker.models.containers import Container
from pandia import BIN_PATH
from pandia.agent.action import Action
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.env_container import ObservationThread
from pandia.agent.observation import Observation
from pandia.agent.reward import reward
from pandia.agent.utils import sample
from pandia.log_analyzer_sender import StreamingContext
from ray import tune
from typing import Optional


class WebRTCEmulatorEnv(gymnasium.Env):
    def __init__(self, client_id=None, rank=None, duration=ENV_CONFIG['duration'], # Exp settings
                 resolution=ENV_CONFIG['width'], fps=ENV_CONFIG['fps'], # Source settings
                 bw=ENV_CONFIG['bandwidth_range'],  # Network settings
                 delay=ENV_CONFIG['delay_range'], loss=ENV_CONFIG['loss_range'], # Network settings
                 action_keys=ENV_CONFIG['action_keys'], # Action settings
                 obs_keys=ENV_CONFIG['observation_keys'], # Observation settings
                 monitor_durations=ENV_CONFIG['observation_durations'], # Observation settings
                 print_step=True, print_period=2, logging_path=None, # Logging settings
                 sb3_logging_path=None, enable_own_logging=False, # Logging settings
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
        self.logging_path = logging_path
        self.sb3_logging_path = sb3_logging_path
        if enable_own_logging:
            self.logging_path = f'/tmp/pandia_{self.uuid}.log'
            self.sb3_logging_path = f'/tmp/sb3_{self.uuid}.log'
        # RL settings
        self.step_duration = step_duration
        self.init_timeout = 10
        self.hisory_size = 1
        # RL state
        self.step_count = 0
        self.context: StreamingContext
        self.obs_keys = list(sorted(obs_keys))
        self.monitor_durations = list(sorted(monitor_durations))
        self.observation: Observation = Observation(self.obs_keys, 
                                                    durations=self.monitor_durations,
                                                    history_size=self.hisory_size)
        # ENV state
        self.receiver_container: Container
        self.termination_ts = 0
        self.action_keys = list(sorted(action_keys))
        self.action_space = Action(action_keys).action_space()
        self.observation_space = \
            Observation(self.obs_keys, self.monitor_durations, self.hisory_size)\
                .observation_space()
        # Tracking
        self.docker_client = docker.from_env()
        self.control_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.obs_socket = self.create_observer()
        self.obs_thread = ObservationThread(self.obs_socket)
        self.obs_thread.start()
        self.start_container()

    def start_container(self):
        cmd = f'docker run -d --rm --network sb3_net --name {self.container_name} '\
              f'--hostname {self.container_name} '\
              f'--cap-add=NET_ADMIN --env NVIDIA_DRIVER_CAPABILITIES=all '\
              f'--runtime=nvidia --gpus all '\
              f'-v /tmp:/tmp '\
              f'--env PRINT_STEP=True -e SENDER_LOG=/tmp/sender.log --env BANDWIDTH=1000-3000 '\
              f'--env OBS_SOCKET_PATH={self.obs_socket_path} '\
              f'--env LOGGING_PATH={self.logging_path} '\
              f'--env SB3_LOGGING_PATH={self.sb3_logging_path} '\
              f'--env CTRL_SOCKET_PATH={self.ctrl_socket_path} '\
              f'johnson163/pandia_emulator python -um sb3_client'
        print(cmd)
        os.system(cmd)
        self.container = self.docker_client.containers.get(self.container_name) # type: ignore
        ts = time.time()
        while time.time() - ts < 3:
            try:
                self.control_socket.connect(self.ctrl_socket_path)
                break
            except FileNotFoundError:
                time.sleep(0.1)
        if time.time() - ts > 3:
            raise Exception(f'Cannot connect to {self.ctrl_socket_path}')


    def stop_container(self):
        if self.container:
            os.system(f'docker stop {self.container.id} > /dev/null &')

    @property
    def container_name(self):
        return f'sb3_emulator_{self.uuid}'

    @property
    def shm_name(self):
        return f"pandia_{self.uuid}"

    @property
    def ctrl_socket_path(self):
        return f'/tmp/{self.uuid}_ctrl.sock'

    @property
    def obs_socket_path(self):
        return f'/tmp/{self.uuid}_obs.sock'

    def log(self, msg):
        print(f'[{self.uuid}, {time.time() - self.start_ts:.02f}] {msg}', flush=True)

    def sample_net_params(self):
        return {
            'bw': sample(self.bw0),
            'delay': sample(self.delay0),
            'loss': sample(self.loss0),
        }


    def start_webrtc(self):
        config = self.sample_net_params()
        print(f'Starting WebRTC, {config}', flush=True)
        buf = bytearray(1)
        buf[0] = 2
        buf += json.dumps(config).encode()
        self.control_socket.send(buf)

    def stop_webrtc(self):
        print('Stopping WebRTC...', flush=True)
        buf = bytearray(1)
        buf[0] = 0  
        self.control_socket.send(buf)
        
    def create_observer(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.bind(self.obs_socket_path)
        print(f'Listening on IPC socket {self.obs_socket_path}')
        return sock

    def reset(self, seed=None, options=None):
        self.stop_webrtc()
        self.context = StreamingContext(monitor_durations=self.monitor_durations)
        self.obs_thread.context = self.context
        self.termination_ts = 0
        self.step_count = 0
        self.last_print_ts = 0
        self.observation = Observation(self.obs_keys, self.monitor_durations, self.hisory_size)
        self.start_ts = time.time()
        return self.observation.array(), {}

    def close(self):
        self.stop_container()
        if os.path.exists(self.obs_socket_path):
            os.remove(self.obs_socket_path)
        self.obs_thread.stop()

    def step(self, action: np.ndarray):
        self.context.reset_step_context()
        act = Action.from_array(action, self.action_keys)
        buf = bytearray(Action.shm_size() + 1)
        act.write(buf)
        buf[1:] = buf[:-1]
        buf[0] = 1
        self.control_socket.send(buf)

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


tune.register_env('WebRTCEmulatorEnv', lambda config: WebRTCEmulatorEnv(**config))
gymnasium.register('WebRTCEmulatorEnv', entry_point='pandia.agent.env_emulator:WebRTCEmulatorEnv', 
                   nondeterministic=True)


def test():
    num_envs = 1
    episodes = 1
    duration = 300000

    if num_envs == 1:
        envs = gymnasium.make("WebRTCEmulatorEnv", bw=2000, delay=5, duration=duration,
                              logging_path='/tmp/pandia.log', sb3_logging_path='/tmp/sb3.log')
    else:
        envs = gymnasium.vector.make("WebRTCEmulatorEnv", num_envs=num_envs, bw=2000, duration=30)
    action = Action(ENV_CONFIG['action_keys'])
    action.bitrate = 1000 
    action.pacing_rate = 2048000
    actions = [action.array()] * num_envs if num_envs > 1 else action.array()
    try:
        for _ in range(episodes):
            envs.reset()
            while True:
                _, _, terminated, truncated, _ = envs.step(actions)
                if np.any(terminated) or np.any(truncated):
                    break
    except KeyboardInterrupt:
        pass
    envs.close()


if __name__ == '__main__':
    test()
