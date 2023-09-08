import os
import socket
import subprocess
import time
import uuid
import docker
import threading
import gymnasium
import numpy as np
from pandia.agent.action import Action
from pandia.agent.env_client import ActionHistory
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.observation import Observation
from pandia.agent.reward import reward
from pandia.constants import WEBRTC_SENDER_SB3_PORT
from pandia.log_analyzer_sender import StreamingContext
from ray import tune
from typing import Optional
from docker.models.containers import Container


NETWORK = None
STUN: Container


def setup_containers():
    global NETWORK
    global STUN
    client = docker.from_env()
    try:
        NETWORK = client.networks.get('sb3_net')
    except docker.errors.NotFound: # type: ignore
        NETWORK = client.networks.create('sb3_net', driver='bridge')
    try:
        STUN = client.containers.get('sb3_stun') # type: ignore
    except Exception: # type: ignore
        os.system(f'docker run -d --rm --network sb3_net --name sb3_stun ich777/stun-turn-server')
        STUN = client.containers.get('sb3_stun') # type: ignore


class WebRTContainerEnv(gymnasium.Env):
    def __init__(self, client_id=None, rank=None, duration=ENV_CONFIG['duration'], # Exp settings
                 width=ENV_CONFIG['width'], fps=ENV_CONFIG['fps'], # Source settings
                 bw=ENV_CONFIG['bandwidth_range'],  # Network settings
                 delay=ENV_CONFIG['delay_range'], loss=ENV_CONFIG['loss_range'], # Network settings
                 action_keys=ENV_CONFIG['action_keys'], # Action settings
                 obs_keys=ENV_CONFIG['observation_keys'], # Observation settings
                 monitor_durations=ENV_CONFIG['observation_durations'], # Observation settings
                 print_step=False, print_period=1,# Logging settings
                 step_duration=ENV_CONFIG['step_duration'], # RL settings
                 termination_timeout=ENV_CONFIG['termination_timeout'] # Exp settings
                 ) -> None:
        super().__init__()
        print(f'Creating WebRTCSb3Env with client_id={client_id}, rank={rank}')
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
        self.print_period = print_period
        self.stop_event: Optional[threading.Event] = None
        self.logging_buf = []
        # RL settings
        self.step_duration = step_duration
        self.init_timeout = 30
        self.hisory_size = 1
        # RL state
        self.step_count = 0
        self.context: StreamingContext
        self.obs_keys = list(sorted(obs_keys))
        self.monitor_durations = list(sorted(monitor_durations))
        self.observation: Observation = Observation(self.obs_keys, durations=self.monitor_durations,
                                                    history_size=self.hisory_size)
        self.process_sender: Optional[subprocess.Popen] = None
        # ENV state
        self.receiver_container: Container
        self.sender_container: Container
        self.container_socket: socket.socket = None
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
        self.docker_client = docker.from_env()

        self.start_containers()

    def create_listen_socket(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        port = 0
        for i in range(100):
            try:
                port = 9990 - i
                sock.bind(('0.0.0.0', port))
                break
            except OSError as e:
                print(e)
        print(f'Listening on port {port}')
        return sock, port

    def start_containers(self):
        cid = str(uuid.uuid4())[:8]
        print('cid: ', cid)
        os.system(f'docker run -d --rm --network sb3_net --name sb3_receiver_{cid} --hostname sb3_receiver_{cid} '
                  f'--runtime=nvidia --gpus all '
                  f'--cap-add=NET_ADMIN -e NVIDIA_DRIVER_CAPABILITIES=all '
                  f'-v /tmp/pandia:/tmp '
                  f'--env STUN_NAME=sb3_stun johnson163/pandia_receiver')
        self.receiver_container = self.docker_client.containers.get(f'sb3_receiver_{cid}') # type: ignore
        self.receiver_container_ip = self.receiver_container.attrs['NetworkSettings']['Networks']['sb3_net']['IPAddress']
        cmd = f'docker run -d --rm --network sb3_net --name sb3_sender_{cid} --hostname sb3_sender_{cid} '\
              f'--cap-add=NET_ADMIN --env NVIDIA_DRIVER_CAPABILITIES=all '\
              f'--runtime=nvidia --gpus all '\
              f'-v /tmp/pandia:/tmp '\
              f'--env RECEIVER_IP={self.receiver_container_ip} '\
              f'--env PRINT_STEP=True -e SENDER_LOG=/tmp/sender.log --env BANDWIDTH=1000-3000 '\
              f'johnson163/pandia_sender python -um sb3_client'
        os.system(cmd)
        self.sender_container = self.docker_client.containers.get(f'sb3_sender_{cid}') # type: ignore
        self.sender_container_ip = self.sender_container.attrs['NetworkSettings']['Networks']['sb3_net']['IPAddress']
        print(f'Sender container IP: {self.sender_container_ip}')
        self.container_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.container_socket.connect((self.sender_container_ip, WEBRTC_SENDER_SB3_PORT))
        time.sleep(1)

    def stop_containers(self):
        print('Stopping containers...', flush=True)
        if self.sender_container:
            os.system(f'docker stop {self.sender_container.id}')
        if self.receiver_container:
            os.system(f'docker stop {self.receiver_container.id}')

    def start_webrtc(self):
        print('Starting WebRTC...', flush=True)
        buf = bytearray(1)
        buf[0] = 0  
        self.container_socket.send(buf)

    def reset(self, seed=None, options=None):
        self.context = StreamingContext(monitor_durations=self.monitor_durations)
        self.action_history = ActionHistory()
        self.termination_ts = 0
        self.step_count = 0
        self.observation = Observation(self.obs_keys, self.monitor_durations, self.hisory_size)
        self.start_ts = time.time()
        return self.observation.array(), {}

    def close(self):
        self.stop_containers()

    def step(self, action: np.ndarray):
        self.context.reset_step_context()
        act = Action.from_array(action, self.action_keys)
        self.action_history.append(act)
        buf = bytearray(Action.shm_size() + 1)
        buf[0] = 1
        act.write(buf[1:])
        self.container_socket.send(buf)
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
                print(f'#{self.step_count}@{int((time.time() - self.start_ts))}s '
                      f'R.w.: {r:.02f}, Act.: {act}Obs.: {self.observation}', flush=True)
        self.step_count += 1
        terminated = self.termination_ts > 0 and self.context.last_ts > self.termination_ts
        return self.observation.array(), r, terminated, truncated, {}


setup_containers()
tune.register_env('WebRTContainerEnv', lambda config: WebRTContainerEnv(**config))
gymnasium.register('WebRTContainerEnv', entry_point='pandia.agent.env_container:WebRTContainerEnv', nondeterministic=True)


if __name__ == '__main__':
    env = WebRTContainerEnv()
    env.reset()
    ts = time.time()
    try:
        while time.time() - ts < 10:
            env.step(env.action_space.sample())
    except KeyboardInterrupt:
        pass
    env.close()
