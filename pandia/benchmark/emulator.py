import json
import os
from typing import List
from matplotlib import pyplot as plt

import numpy as np
from pandia import RESULTS_PATH
from pandia.agent.action import Action
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.env_emulator import WebRTCEmulatorEnv
from pandia.agent.observation import Observation
from pandia.analysis.stream_illustrator import DPI, FIG_EXTENSION, generate_diagrams, setup_diagrams_path
from pandia.constants import K, M
from pandia.context.frame_context import FrameContext


def run(env, br, config, bw):
    action = Action(config['action_keys'])
    action.bitrate = br
    rewards = []
    frame_bitrates = []
    g2g = []
    actions = []
    env.reset()
    while True:
        obs, reward, terminated, truncated, _ = env.step(action.array())
        obs = Observation.from_array(obs)
        actions.append(float(action.bitrate))
        rewards.append(float(reward))
        frame_bitrates.append(obs.get_data(obs.data[0][0], 'frame_bitrate', numeric=True))
        g2g.append(float(obs.get_data(obs.data[0][0], 'frame_decoded_delay', numeric=True)))
        if terminated or truncated:
            break
    print(f'===BW: {env.net_sample["bw"] / M:.02f} mbps, bitrate: {br / M:.02f} mbps, '
          f'G2G delay: {np.mean(g2g) * 1000:.02f} ms, '
          f'reward: {np.mean(rewards):.02f}, '
          f'frame bitrate: {np.mean(frame_bitrates) / M:.02f} mbps===')
    return rewards, actions, g2g


def draw_diagrams(env: WebRTCEmulatorEnv, data, draw_ctx=False):
    path = os.path.join(RESULTS_PATH, "benchmark_emulator")
    setup_diagrams_path(path)
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
    bw = 3 * M
    config = ENV_CONFIG
    config['network_setting']['bandwidth'] = bw
    config['network_setting']['delay'] = .005
    config['gym_setting']['print_step'] = True
    config['gym_setting']['print_period'] = 1
    config['gym_setting']['duration'] = 10
    config['gym_setting']['skip_slow_start'] = 1
    config['gym_setting']['step_duration'] = .1
    # config['gym_setting']['logging_path'] = '/tmp/pandia.log'
    config['gym_setting']['sb3_logging_path'] = '/tmp/pandia.log'
    env = WebRTCEmulatorEnv(config=config, curriculum_level=None) # type: ignore
    data = []
    # br_list = list(range(100 * K, 1000 * K, 200 * K)) + list(range(1 * M, 10 * M, 2 * M))
    # br_list = list(range(100 * K, 1000 * K, 100 * K)) 
    buckets = 20
    br_list = (np.arange(20) + 1) / 20 * 4 * M
    json_data = []
    for br in br_list:
        rewards, bitrates, delays = run(env, br, config, bw)
        frames: List[FrameContext] = list(env.context.frames.values())
        loss_rate = (1 - len([f for f in frames if f.decoded_at > 0]) / len(frames)) if len(frames) > 0 else 1
        g2g_delay = np.mean([f.g2g_delay() for f in frames if f.decoded_at > 0])
        data.append((br, loss_rate, g2g_delay, np.mean(rewards)))
        json_data.append({'bitrate': br, 'rewards': rewards, 'delays': delays})
    env.close()

    path = os.path.join(os.path.join(RESULTS_PATH, 'benchmark_emulator'), 'bitrate_vs_reward.json')
    with open(path, 'w') as f: 
        json.dump(json_data, f)
    json_str = json.dumps(json_data)
    path2 = '/tmp/bitrate_vs_reward.json'
    with open(path2, 'w') as f:
        f.write(json_str)
    draw_diagrams(env, data)


def process_data():
    with open(os.path.join(RESULTS_PATH, 'benchmark_emulator/bitrate_vs_reward.json'), 'r') as f: 
        data = json.load(f)
        x = []
        y_delay = []
        y_reward = []
        for d in data:
            bitrate = d['bitrate']
            rewards = d['rewards']
            delays = d['delays']
            x.append(bitrate / M)
            y_delay.append([np.percentile(delays, p) * 1000 for p in [10, 50, 90]])
            y_reward.append([np.percentile(rewards, p) for p in [10, 50, 90]])

        yerr_reward = [y[1] - y[0] for y in y_reward], [y[2] - y[1] for y in y_reward]
        path = os.path.join(RESULTS_PATH, 'benchmark_emulator')
        plt.close()
        fig = plt.gcf()
        fig.set_size_inches(3, 1)
        plt.xlabel('Bitrate action (Mbps)')
        plt.ylabel('Reward')
        # plt.errorbar(x, [y[1] for y in y_reward], yerr=yerr_reward)
        plt.plot(x, [y[1] for y in y_reward])
        plt.tight_layout(pad=.1)
        plt.savefig(os.path.join(path, f'bitrate_vs_reward0.{FIG_EXTENSION}'), dpi=DPI)

        # yerr_delay = [y[1] - y[0] for y in y_delay], [y[2] - y[1] for y in y_delay]
        # path = os.path.join(RESULTS_PATH, 'benchmark_emulator')
        # plt.close()
        # plt.xlabel('Bitrate action (Mbps)')
        # plt.ylabel('Delay (ms)')
        # plt.errorbar(x, [y[1] for y in y_delay], yerr=yerr_delay)
        # plt.savefig(os.path.join(path, f'bitrate_vs_g2g.{FIG_EXTENSION}'), dpi=DPI)


if __name__ == "__main__":
    # main()
    process_data()

