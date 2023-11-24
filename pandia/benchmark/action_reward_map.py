import os
import gymnasium
from matplotlib import pyplot as plt
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO, SAC
from pandia import RESULTS_PATH, SB3_LOG_PATH
from pandia.agent.action import Action
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.env_simple_simulator import WebRTCSimpleSimulatorEnv
from pandia.constants import M
from pandia.agent.observation import Observation
from pandia.log_analyzer_sender import analyze_stream


def main():
    buckets = 10
    data = []
    config = ENV_CONFIG
    config['network_setting']['bandwidth'] = 4 * M
    config['network_setting']['delay'] = .01
    env = gymnasium.make("WebRTCSimpleSimulatorEnv", config=config, curriculum_level=None)
    br = ENV_CONFIG['action_limit']['bitrate']
    for bucket in range(buckets):
        bitrate = int(br[0] + (br[1] - br[0]) * bucket / buckets)
        print(f'Bitrate: {bitrate/M:.02f}Mbps')
        action = Action(ENV_CONFIG['action_keys'])
        action.bitrate = bitrate
        action.pacing_rate = 1000 * M
        env.reset()
        rewards = []
        obs_data = []
        while True:
            obs, reward, terminated, truncated, _ = env.step(action.array())
            observation = Observation.from_array(obs)
            rewards.append(reward)
            obs_data.append([
                action.bitrate / M,
                observation.get_data(observation.data[0][0], 'frame_decoded_delay', numeric=True) * 1000, 
                observation.get_data(observation.data[0][0], 'frame_bitrate', numeric=True) / M,
                ])
            if terminated or truncated:
                break
        data.append((action.bitrate / M, np.mean(rewards)))
    plt.plot([d[0] for d in data], [d[1] for d in data])
    plt.xlabel('Bitrate (Mbps)')
    plt.ylabel('Reward')
    output_dir = os.path.join(RESULTS_PATH, 'benchmark')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'action_test_bitrate_reward.pdf'))

    plt.close()
    fig, ax1 = plt.subplots()
    ax1.plot(np.arange(len(obs_data)), [d[0] for d in obs_data])
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Bitrate (mbps)')
    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(obs_data)), [d[1] for d in obs_data])
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Frame decoded delay (ms)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'action_test_bitrate_delay.pdf'))


if __name__ == '__main__':
    main()
