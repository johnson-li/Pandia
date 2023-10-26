import os
import gymnasium
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from pandia import RESULTS_PATH, SB3_LOG_PATH
from pandia.agent.action import Action
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.env_simple_simulator import WebRTCSimpleSimulatorEnv
from pandia.agent.observation import Observation
from pandia.constants import M
from pandia.log_analyzer_sender import analyze_stream


def model_path():
    path = '/tmp/WebRTCSimpleSimulatorEnv'
    print(f'Loading model from {path}')
    return os.path.join(path, 'best_model.zip')


def main_multi_env():
    for bw in range(bw0, bw0 + 1):
        env.bw0 = bw / 10 * 1024
        print(f'Eval with bw: {env.bw0 / 1024:.02f} Mbps')
        obs, _ = env.reset()
        obs, _ = env.reset()
        rewards = []
        delays = []
        actions = []
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            obs_obj = env.observation
            act_obj = Action.from_array(action, env.action_keys)
            actions.append(act_obj.bitrate)
            delays.append(obs_obj.get_data(obs_obj.data[0][0], 'frame_decoded_delay'))
            rewards.append(reward)
            if terminated or truncated:
                break
        data.append((env.bw0, np.mean(actions), np.mean(delays), np.mean(rewards)))
        print(f'bw: {env.bw0 / 1024:.02f}Mbps, bitrate: {np.mean(actions) / 1024:.02f}Mbps, '
              f'delay: {np.mean(delays):.02f}ms, reward: {np.mean(rewards):.02f}')


def main_single_env():
    duration = 30
    bw = 4500
    env = WebRTCSimpleSimulatorEnv(duration=duration, delay=20, print_step=False)
    model = PPO.load(model_path(), env)
    data = []
    env.bw0 = bw 
    print(f'Eval with bw: {env.bw0 / 1024:.02f} Mbps')
    obs, _ = env.reset()
    obs, _ = env.reset()
    rewards = []
    delays = []
    actions = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        obs_obj = env.observation
        act_obj = Action.from_array(action, env.action_keys)
        actions.append(act_obj.bitrate)
        delays.append(obs_obj.get_data(obs_obj.data[0][0], 'frame_decoded_delay'))
        rewards.append(reward)
        if terminated or truncated:
            break
    data.append((env.bw0, np.mean(actions), np.mean(delays), np.mean(rewards)))
    print(f'bw: {env.bw0 / 1024:.02f}Mbps, bitrate: {np.mean(actions) / 1024:.02f}Mbps, '
            f'delay: {np.mean(delays):.02f}ms, reward: {np.mean(rewards):.02f}')
    env.close()

    # Plot evaluation results
    x = np.arange(len(actions)) 
    plt.close()
    fig, ax1 = plt.subplots()
    ax1.plot(x, actions, 'r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Bitrate action (Mbps)')
    ax2 = ax1.twinx()
    ax2.plot(x, delays, 'b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.set_ylabel('G2G delay (ms)')
    plt.tight_layout()
    output_dir = os.path.join(RESULTS_PATH, 'eval_sb3')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'single_run_bitrate_delay.pdf'))

    plt.close()
    fig, ax1 = plt.subplots()
    ax1.plot(x, actions, 'r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Bitrate action (Mbps)')
    ax2 = ax1.twinx()
    ax2.plot(x, rewards, 'b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.set_ylabel('Reward')
    plt.tight_layout()
    output_dir = os.path.join(RESULTS_PATH, 'eval_sb3')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'single_run_bitrate_reward.pdf'))


def test_action_rewards():
    buckets = 40
    data = []
    config = ENV_CONFIG
    config['network_setting']['bandwidth'] = 20 * M
    config['network_setting']['delay'] = .01
    env = gymnasium.make("WebRTCSimpleSimulatorEnv", config=config)
    br = ENV_CONFIG['training_setting']['bitrate_range']
    # br = [1 * M, 40 * M]
    for bucket in range(buckets):
        bitrate = int(br[0] + (br[1] - br[0]) * bucket / buckets)
        print(f'bitrate: {bitrate/M:.02f}Mbps')
        action = Action(ENV_CONFIG['action_keys'])
        action.bitrate = bitrate
        action.pacing_rate = 1000 * M
        env.reset()
        rewards = []
        while True:
            _, reward, terminated, truncated, _ = env.step(action.array())
            rewards.append(reward)
            if terminated or truncated:
                break
        # print(f'bw: {bw:.02f}Mbps, bitrate: {bitrate/1024:.02f}Mbps, reward: {np.mean(rewards):.02f}')
        data.append((action.bitrate / M, np.mean(rewards)))
    plt.plot([d[0] for d in data], [d[1] for d in data])
    plt.xlabel('Bitrate (Mbps)')
    plt.ylabel('Reward')
    output_dir = os.path.join(RESULTS_PATH, 'eval_sb3')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'action_test_bitrate_reward.pdf'))


if __name__ == "__main__":
    # main_single_env()
    test_action_rewards()
