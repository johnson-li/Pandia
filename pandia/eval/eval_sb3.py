import argparse
import os
import time
import gymnasium
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from pandia import RESULTS_PATH, SB3_LOG_PATH
from pandia.agent.action import Action
from pandia.agent.env_client import WebRTCEnv0
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.env_simple_simulator import WebRTCSimpleSimulatorEnv
from pandia.agent.observation import Observation
from pandia.log_analyzer_sender import analyze_stream


def model_path():
    # path = os.path.expanduser('~/sb3_logs/ppo')
    # dirs = [f for f in os.listdir(path) if f.startswith('WebRTCSimpleSimulatorEnv')]
    # dirs = sorted(dirs, key=lambda x: int(x.split('_')[-1]))
    # path = os.path.join(path, dirs[-1])
    path = '/tmp/WebRTCSimpleSimulatorEnv'
    print(f'Loading model from {path}')
    return os.path.join(path, 'best_model.zip')


def main_trained():
    duration = 30
    env = WebRTCSimpleSimulatorEnv(duration=duration, delay=0)
    model = PPO.load(model_path(), env)
    data = []
    for bw in range(1, 25):
        env.bw0 = bw / 10 * 1024
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
        print(f'bw: {env.bw0 / 1024:.02f}Mbps, bitrate: {np.mean(actions):.02f}Mbps, delay: {np.mean(delays):.02f}ms, reward: {np.mean(rewards):.02f}')
    env.close()

    # Plot evaluation results
    # x = [d[0] for d in data]
    # y1 = [d[1] for d in data]
    # y2 = [d[2] for d in data]
    # plt.close()
    # fig, ax1 = plt.subplots()
    # ax1.plot(x, y1, 'r')
    # ax1.tick_params(axis='y', labelcolor='r')
    # ax1.set_xlabel('Step')
    # ax1.set_ylabel('Bitrate setting (Mbps)')
    # ax2 = ax1.twinx()
    # ax2.plot(x, y2, 'b')
    # ax2.tick_params(axis='y', labelcolor='b')
    # ax2.set_ylabel('G2G delay (ms)')
    # ax2.set_ylim(0, 100)
    # plt.tight_layout()
    # output_dir = os.path.join(RESULTS_PATH, 'eval_sb3')
    # os.makedirs(output_dir, exist_ok=True)
    # plt.savefig(os.path.join(output_dir, 'bitrate_delay.pdf'))


def test_action_rewards():
    data = []
    for bw in [3]:
        for bitrate in range(10, 31):
            env = gymnasium.make("WebRTCSimpleSimulatorEnv", 
                                bw=bw * 1024, delay=0, print_step=False)
            action = Action(ENV_CONFIG['action_keys'])
            action.bitrate = int(bitrate / 10 * 1024)
            action.pacing_rate = 1000 * 1024
            env.reset()
            rewards = []
            while True:
                _, reward, terminated, truncated, _ = env.step(action.array())
                rewards.append(reward)
                if terminated or truncated:
                    break
            # print(f'bw: {bw:.02f}Mbps, bitrate: {bitrate:.02f}Mbps, reward: {np.mean(rewards):.02f}')
            data.append((bitrate, np.mean(rewards)))
    plt.plot([d[0] for d in data], [d[1] for d in data])
    plt.xlabel('Bitrate (Mbps)')
    plt.ylabel('Reward')
    output_dir = os.path.join(RESULTS_PATH, 'eval_sb3')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'bitrate_reward.pdf'))


if __name__ == "__main__":
    main_trained()
    # test_action_rewards()
