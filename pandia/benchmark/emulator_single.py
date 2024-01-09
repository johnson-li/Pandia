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
    config['network_setting']['delay'] = .001
    config['gym_setting']['print_step'] = True
    config['gym_setting']['print_period'] = 0
    config['gym_setting']['duration'] = 10000
    config['gym_setting']['skip_slow_start'] = 1
    config['gym_setting']['step_duration'] = .1
    # config['gym_setting']['logging_path'] = '/tmp/pandia.log'
    config['gym_setting']['sb3_logging_path'] = '/tmp/pandia.log'
    env = WebRTCEmulatorEnv(config=config, curriculum_level=None) # type: ignore
    action = Action(config['action_keys'])
    rewards = []
    delays = []
    actions = []
    pd = 50
    br_list = [bw / 3] * pd + [bw / 3 * 2] * pd + [bw] * pd + [bw / 3 * 2] * pd + [bw / 3] * pd 
    try:
        env.reset()
        for br in br_list:
            action.bitrate = br
            obs, reward, terminated, truncated, _ = env.step(action.array())
            obs = Observation.from_array(obs)
            actions.append(action.bitrate)
            delays.append(obs.get_data(obs.data[0][0], 'frame_decoded_delay', numeric=True))
            rewards.append(reward)
            # frames: List[FrameContext] = list(env.context.frames.values())
            # loss_rate = (1 - len([f for f in frames if f.decoded_at > 0]) / len(frames)) if len(frames) > 0 else 1
            # g2g_delay = np.mean([f.g2g_delay() for f in frames if f.decoded_at > 0])
    except Exception as e:
        pass
    finally:
        env.close()

    rewards = np.array(rewards)
    delays = np.array(delays)
    actions = np.array(actions)
    
    path = os.path.join(RESULTS_PATH, "benchmark_emulator_single")
    generate_diagrams(path, env.context)

    plt.close()
    fig, ax1 = plt.subplots()
    color = 'r'
    ax1.plot(np.arange(len(actions)), actions / M, color)
    ax1.set_ylabel('Action Bitrate (Mbps)')
    ax1.set_xlabel('Step')
    ax1.spines['left'].set_color(color)
    ax1.yaxis.label.set_color(color)
    ax1.tick_params(axis='y', colors=color)
    ax2 = ax1.twinx()
    color = 'b'
    ax2.plot(np.arange(len(rewards)), rewards, color)
    ax2.set_ylabel('Reward')
    ax2.spines['right'].set_color(color)
    ax2.yaxis.label.set_color(color)
    ax2.tick_params(axis='y', colors=color)
    plt.title(f'Bandwidth: {env.net_sample["bw"] / M:.02f} Mbps')
    plt.tight_layout()
    plt.savefig(os.path.join(path, f'bitrate_vs_reward.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    fig, ax1 = plt.subplots()
    color = 'r'
    ax1.plot(np.arange(len(actions)), actions / M, color)
    ax1.set_ylabel('Action Bitrate (Mbps)')
    ax1.set_xlabel('Step')
    ax1.spines['left'].set_color(color)
    ax1.yaxis.label.set_color(color)
    ax1.tick_params(axis='y', colors=color)
    ax2 = ax1.twinx()
    color = 'b'
    ax2.plot(np.arange(len(rewards)), delays * 1000, color)
    ax2.set_ylabel('G2G Delay (ms)')
    ax2.spines['right'].set_color(color)
    ax2.yaxis.label.set_color(color)
    ax2.tick_params(axis='y', colors=color)
    plt.title(f'Bandwidth: {env.net_sample["bw"] / M:.02f} Mbps')
    plt.tight_layout()
    plt.savefig(os.path.join(path, f'bitrate_vs_g2g_delay.{FIG_EXTENSION}'), dpi=DPI)


if __name__ == "__main__":
    main()
