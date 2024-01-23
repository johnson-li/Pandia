import os
import gymnasium
from matplotlib import pyplot as plt
import numpy as np
from pandia import RESULTS_PATH, SB3_LOG_PATH
from pandia.agent.action import Action
from pandia.agent.curriculum_level import CURRICULUM_LEVELS
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.utils import deep_update
from pandia.constants import K, M
from pandia.agent.observation import Observation
from pandia.log_analyzer_sender import analyze_stream


def main():
    buckets = 10
    br = [1 * M, 5 * M]
    data = []
    config = ENV_CONFIG
    deep_update(config, CURRICULUM_LEVELS[2])
    config['network_setting']['bandwidth'] = 3 * M
    config['network_setting']['delay'] = .008
    # config['action_limit']['bitrate'] = br
    env = gymnasium.make("WebRTCSimpleSimulatorEnv", config=config, curriculum_level=None)
    for bucket in range(buckets):
        bitrate = int(br[0] + (br[1] - br[0]) * bucket / buckets)
        action = Action(ENV_CONFIG['action_keys'])
        action.bitrate = bitrate
        action.pacing_rate = 1000 * M
        env.reset()
        bitrates = []
        rewards = []
        delays = []
        obs_data = []
        while True:
            obs, reward, terminated, truncated, _ = env.step(action.array())
            observation = Observation.from_array(obs)
            rewards.append(reward)
            bitrates.append(action.bitrate)
            g2g_delay = observation.get_data(observation.data[0][0], 'frame_decoded_delay', numeric=True) * 1000
            delays.append(g2g_delay)
            frame_bitrate = observation.get_data(observation.data[0][0], 'frame_bitrate', numeric=True) / M # type: ignore
            obs_data.append([action.bitrate / M, g2g_delay, frame_bitrate])
            if terminated or truncated:
                break
        print(f'Bitrate: {bitrate/M:.02f} Mbps, reward: {np.mean(rewards):.02f}')
        print(f'bitrates = [{", ".join(bitrates)}]')
        print(f'rewards = [{", ".join(bitrates)}]')
        print(f'g2g_delay = [{", ".join(bitrates)}]')
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
