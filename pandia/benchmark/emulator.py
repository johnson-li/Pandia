import os
from typing import List
from matplotlib import pyplot as plt

import numpy as np
from pandia import RESULTS_PATH
from pandia.agent.action import Action
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.env_emulator import WebRTCEmulatorEnv
from pandia.agent.observation import Observation
from pandia.analysis.stream_illustrator import DPI, FIG_EXTENSION, generate_diagrams
from pandia.constants import K, M
from pandia.context.frame_context import FrameContext


def run(env, br, config, bw):
    action = Action(config['action_keys'])
    action.bitrate = br
    rewards = []
    frame_bitrates = []
    g2g = []
    env.reset()
    while True:
        obs, reward, terminated, truncated, _ = env.step(action.array())
        obs = Observation.from_array(obs)
        rewards.append(reward)
        frame_bitrates.append(obs.get_data(obs.data[0][0], 'frame_bitrate', numeric=True))
        g2g.append(obs.get_data(obs.data[0][0], 'frame_decoded_delay', numeric=True))
        if terminated or truncated:
            break
    print(f'===BW: {env.net_sample["bw"] / M:.02f} mbps, bitrate: {br / M:.02f} mbps, '
          f'G2G delay: {np.mean(g2g) * 1000:.02f} ms, '
          f'reward: {np.mean(rewards):.02f}, '
          f'frame bitrate: {np.mean(frame_bitrates) / M:.02f} mbps===')
    return rewards


def draw_diagrams(env: WebRTCEmulatorEnv, data, draw_ctx=False):
    path = os.path.join(RESULTS_PATH, "benchmark_emulator")
    if draw_ctx:
        generate_diagrams(path, env.context)

    plt.close()
    plt.plot([d[0] / M for d in data], [d[1] * 100 for d in data])
    plt.xlabel('Bitrate (Mbps)')
    plt.ylabel('Frame loss rate (%)')
    plt.title(f'Bandwidth: {env.net_sample["bw"] / M:.02f} Mbps')
    plt.savefig(os.path.join(path, f'bitrate_vs_frame_loss_rate.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    plt.plot([d[0] / M for d in data], [d[2] * 1000 for d in data])
    plt.xlabel('Bitrate (Mbps)')
    plt.ylabel('G2G delay (ms)')
    plt.title(f'Bandwidth: {env.net_sample["bw"] / M:.02f} Mbps')
    plt.savefig(os.path.join(path, f'bitrate_vs_g2g_delay.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    plt.plot([d[0] / M for d in data], [d[3] for d in data])
    plt.xlabel('Bitrate (Mbps)')
    plt.ylabel('Reward')
    plt.title(f'Bandwidth: {env.net_sample["bw"] / M:.02f} Mbps')
    plt.savefig(os.path.join(path, f'bitrate_vs_reward.{FIG_EXTENSION}'), dpi=DPI)


def main():
    bw = 500 * K
    config = ENV_CONFIG
    config['network_setting']['bandwidth'] = bw
    config['network_setting']['delay'] = .001
    config['gym_setting']['print_step'] = True
    config['gym_setting']['print_period'] = 1
    config['gym_setting']['duration'] = 10
    config['gym_setting']['skip_slow_start'] = 0
    config['gym_setting']['step_duration'] = .1
    # config['gym_setting']['logging_path'] = '/tmp/pandia.log'
    config['gym_setting']['sb3_logging_path'] = '/tmp/pandia.log'
    env = WebRTCEmulatorEnv(config=config, curriculum_level=None) # type: ignore
    data = []
    # br_list = list(range(100 * K, 1000 * K, 100 * K)) + list(range(1 * M, 10 * M, 1 * M))
    br_list = list(range(100 * K, 1000 * K, 100 * K)) 
    try:
        for br in br_list:
            rewards = run(env, br, config, bw)
            frames: List[FrameContext] = list(env.context.frames.values())
            loss_rate = (1 - len([f for f in frames if f.decoded_at > 0]) / len(frames)) if len(frames) > 0 else 1
            g2g_delay = np.mean([f.g2g_delay() for f in frames if f.decoded_at > 0])
            data.append((br, loss_rate, g2g_delay, np.mean(rewards)))
    except Exception as e:
        pass
    finally:
        env.close()
    
    draw_diagrams(env, data)


if __name__ == "__main__":
    main()
