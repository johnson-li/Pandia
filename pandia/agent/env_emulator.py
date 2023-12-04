import json
import os
import socket
import docker
import time
import uuid
import gymnasium
import numpy as np
from pandia import BIN_PATH, RESULTS_PATH
from pandia.agent.action import Action
from pandia.agent.env import WebRTCEnv
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.observation_thread import ObservationThread
from pandia.agent.reward import reward
from pandia.agent.utils import sample
from pandia.analysis.stream_illustrator import illustrate_frame
from pandia.constants import M


class WebRTCEmulatorEnv(WebRTCEnv):
    def __init__(self, config=ENV_CONFIG, curriculum_level=0) -> None: 
        super().__init__(config, curriculum_level)
        # Exp settings
        self.uuid = str(uuid.uuid4())[:8]
        self.termination_timeout = 3
        # Logging settings
        self.logging_path = config['gym_setting'].get('logging_path', None)
        self.sb3_logging_path = config['gym_setting'].get('sb3_logging_path', None)
        self.obs_logging_path = config['gym_setting'].get('obs_logging_path', None)
        self.enable_own_logging = config['gym_setting'].get('enable_own_logging', False)
        if self.enable_own_logging:
            self.logging_path = f'{self.logging_path}.{self.uuid}'
            self.sb3_logging_path = f'{self.sb3_logging_path}.{self.uuid}'
        # RL settings
        self.init_timeout = 10
        # Tracking
        self.docker_client = docker.from_env()
        self.control_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.obs_socket = self.create_observer()
        self.obs_thread = ObservationThread(self.obs_socket, logging_path=self.obs_logging_path)
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
    def ctrl_socket_path(self):
        return f'/tmp/{self.uuid}_ctrl.sock'

    @property
    def obs_socket_path(self):
        return f'/tmp/{self.uuid}_obs.sock'

    def log(self, msg):
        print(f'[{self.uuid}, {time.time() - self.start_ts:.02f}] {msg}', flush=True)

    def start_webrtc(self):
        self.sample_net_params()
        print(f'Starting WebRTC, {self.net_sample}', flush=True)
        buf = bytearray(1)
        buf[0] = 2
        buf += json.dumps(self.net_sample).encode()
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
        ans = super().reset(seed, options)
        self.stop_webrtc()
        self.obs_thread.context = self.context
        self.last_print_ts = 0
        self.start_ts = time.time()
        return ans

    def close(self):
        self.stop_container()
        if os.path.exists(self.obs_socket_path):
            os.remove(self.obs_socket_path)
        self.obs_thread.stop()

    def step(self, action: np.ndarray):
        limit = Action.action_limit(self.action_keys, limit=self.action_limit)
        action = np.clip(action, limit.low, limit.high)
        self.context.reset_step_context()
        act = Action.from_array(action, self.action_keys)
        self.actions.append(act)
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
        r = reward(self.context, self.net_sample)

        if self.print_step and time.time() - self.last_print_ts > self.print_period:
            self.last_print_ts = time.time()
            self.log(f'#{self.step_count}@{int((time.time() - self.start_ts))}s '
                    f'R.w.: {r:.02f}, Act.: {act}, Obs.: {self.observation}')
        self.step_count += 1
        return self.observation.array(), r, False, truncated, {}


gymnasium.register('WebRTCEmulatorEnv', entry_point='pandia.agent.env_emulator:WebRTCEmulatorEnv', 
                   nondeterministic=False)


def test_single():
    episodes = 1
    config = ENV_CONFIG
    config['network_setting']['bandwidth'] = 10 * M
    config['gym_setting']['print_step'] = True
    config['gym_setting']['print_period'] = 1
    config['gym_setting']['duration'] = 30
    config['gym_setting']['step_duration'] = .1
    config['gym_setting']['logging_path'] = '/tmp/pandia.log'
    env: WebRTCEmulatorEnv = gymnasium.make("WebRTCEmulatorEnv", config=config) # type: ignore
    action = Action(config['action_keys'])
    action.bitrate = 8 * M
    try:
        for _ in range(episodes):
            env.reset()
            while True:
                _, _, terminated, truncated, _ = env.step(action.array())
                if np.any(terminated) or np.any(truncated):
                    break
    except KeyboardInterrupt:
        pass
    env.close()
    fig_path = os.path.join(RESULTS_PATH, 'env_emulator_test')
    illustrate_frame(fig_path, env.context)


if __name__ == '__main__':
    test_single()
