import os
import gymnasium
from matplotlib import pyplot as plt
import numpy as np
from pandia import RESULTS_PATH
from pandia.agent.action import Action
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.observation import Observation
from pandia.constants import M
from pandia.agent.env_simple_simulator import WebRTCSimpleSimulatorEnv


def main():
    config = ENV_CONFIG
    config['network_setting']['bandwidth'] = 8 * M
    config['network_setting']['delay'] = .01
    env = gymnasium.make("WebRTCSimpleSimulatorEnv", config=config, curriculum_level=None)
    bitrate = 7.5 * M
    action = Action(ENV_CONFIG['action_keys'])
    action.bitrate = bitrate
    action.pacing_rate = 1000 * M
    env.reset()
    actions = []
    rewards = []
    delays = []
    frame_bitrates = []
    while True:
        obs, reward, terminated, truncated, _ = env.step(action.array())
        observation = Observation.from_array(obs)
        actions.append(action.bitrate / M)
        rewards.append(reward)
        delays.append(observation.get_data(observation.data[0][0], 
                                           'frame_decoded_delay', numeric=True) * 1000 )
        frame_bitrates.append(observation.get_data(observation.data[0][0],
                                                   'frame_bitrate', numeric=True) / M)
        if terminated or truncated:
            break

    plt.close()
    fig, ax1 = plt.subplots()
    ax1.plot(np.arange(len(actions)), actions)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Bitrate (mbps)')
    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(rewards)), rewards)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Reward')
    plt.tight_layout()
    output_dir = os.path.join(RESULTS_PATH, 'benchmark')
    plt.savefig(os.path.join(output_dir, 'action_reward_history.pdf'))

    plt.close()
    fig, ax1 = plt.subplots()
    ax1.plot(np.arange(len(actions)), actions)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Bitrate (mbps)')
    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(delays)), delays)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('G2G delay (ms)')
    plt.tight_layout()
    output_dir = os.path.join(RESULTS_PATH, 'benchmark')
    plt.savefig(os.path.join(output_dir, 'action_delay_history.pdf'))

    plt.close()
    fig, ax1 = plt.subplots()
    ax1.plot(np.arange(len(actions)), actions)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Bitrate (mbps)')
    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(frame_bitrates)), frame_bitrates)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Frame bitrate (mbps)')
    plt.tight_layout()
    output_dir = os.path.join(RESULTS_PATH, 'benchmark')
    plt.savefig(os.path.join(output_dir, 'action_frame_bitrate_history.pdf'))


if __name__ == "__main__":
    main()
