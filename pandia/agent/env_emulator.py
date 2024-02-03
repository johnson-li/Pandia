import json
import os
import socket
import docker
import time
import uuid
import gymnasium
from matplotlib import pyplot as plt
import numpy as np
from pandia import BIN_PATH, RESULTS_PATH
from pandia.agent.action import Action
from pandia.agent.env import WebRTCEnv
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.observation_thread import ObservationThread
from pandia.agent.reward import reward
from pandia.analysis.stream_illustrator import DPI, FIG_EXTENSION, generate_diagrams
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
        self.bad_reward_count = 0
        self.docker_client = docker.from_env()
        self.control_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.obs_socket = self.create_observer()
        self.obs_thread = ObservationThread(self.obs_socket, logging_path=self.obs_logging_path)
        self.obs_thread.start()
        self.start_container()

    def start_container(self):
        cmd = f'docker run -d --rm --name {self.container_name} '\
              f'--hostname {self.container_name} '\
              f'--cap-add=NET_ADMIN --env NVIDIA_DRIVER_CAPABILITIES=all '\
              f'--runtime=nvidia --gpus all '\
              f'-v /tmp:/tmp '\
              f'--env PRINT_STEP=True -e SENDER_LOG=/tmp/sender.log --env BANDWIDTH=1000-3000 '\
              f'--env NVENC=1 --env NVDEC=1 '\
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
        if time.time() - ts > 5:
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
        self.bad_reward_count = 0
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
        action_capped = False
        if self.action_cap:
            # Avoid over-sending by limiting the bitrate to the network bandwidth
            if act.bitrate > self.net_sample['bw'] * self.action_cap:
                act.bitrate = self.net_sample['bw'] * self.action_cap
                action_capped = True  
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
                return self.step(act.array())
            time.sleep(.1)
        ts = time.time()
        if self.step_count == 0:
            self.start_ts = ts
            print(f'WebRTC is running.', flush=True)
        time.sleep(self.step_duration)
        if self.step_count == 0:
            time.sleep(self.skip_slow_start)

        for mb in self.context.monitor_blocks.values():
            mb.update_ts(time.time() - self.obs_thread.ts_offset)
        self.observation.append(self.context.monitor_blocks)
        r = reward(self.context, self.net_sample)

        if self.print_step and time.time() - self.last_print_ts > self.print_period:
            self.last_print_ts = time.time()
            self.log(f'#{self.step_count}@{int((time.time() - self.start_ts))}s, '
                    f'R.w.: {r:.02f}, '
                    f'bw.: {self.net_sample["bw"] / M:.02f} Mbps, '
                    f'Act.: {act}{"(C)" if action_capped else ""}, '
                    f'Obs.: {self.observation}')
        self.step_count += 1
        if r <= -10:
            self.bad_reward_count += 1
        else:
            self.bad_reward_count = 0
        terminated = self.bad_reward_count > 1000
        truncated = self.step_count > self.step_limit
        return self.observation.array(), r, terminated, truncated, {'action': act.array()}


gymnasium.register('WebRTCEmulatorEnv', entry_point='pandia.agent.env_emulator:WebRTCEmulatorEnv', 
                   nondeterministic=False)


def test_single():
    bw = 3 * M
    config = ENV_CONFIG
    config['network_setting']['bandwidth'] = bw
    config['gym_setting']['print_step'] = True
    config['gym_setting']['print_period'] = 1
    config['gym_setting']['duration'] = 1000
    config['gym_setting']['step_duration'] = .1
    config['gym_setting']['logging_path'] = '/tmp/pandia.log'
    config['gym_setting']['skip_slow_start'] = 0
    env: WebRTCEmulatorEnv = gymnasium.make("WebRTCEmulatorEnv", config=config, curriculum_level=None) # type: ignore
    action = Action(config['action_keys'])
    actions = []
    rewards = []
    try:
        env.reset()
        pd = 50
        bitrates = [1 * M] * pd + [2 * M] * pd + [3 * M] * pd + [2 * M] * pd + [1 * M] * pd 
        for bitrate in bitrates:
            action.bitrate = bitrate 
            actions.append(action.bitrate / M)
            _, reward, terminated, truncated, _ = env.step(action.array())
            rewards.append(reward)
    except KeyboardInterrupt:
        pass
    env.close()
    fig_path = os.path.join(RESULTS_PATH, 'env_emulator_test')
    generate_diagrams(fig_path, env.context)

    plt.close()
    fig, ax1 = plt.subplots()
    ax1.plot(np.arange(len(actions)), actions, 'r')
    ax1.set_ylabel('Bitrate (Mbps)')
    ax1.set_xlabel('Step')
    ax1.spines['left'].set_color('r')
    ax1.yaxis.label.set_color('r')
    ax1.tick_params(axis='y', colors='r')
    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(rewards)), rewards, 'b')
    ax2.set_ylabel('Reward')
    ax2.spines['right'].set_color('b')
    ax2.yaxis.label.set_color('b')
    ax2.tick_params(axis='y', colors='b')
    plt.savefig(os.path.join(fig_path, f'bitrate_reward.{FIG_EXTENSION}'), dpi=DPI)


if __name__ == '__main__':
    test_single()
