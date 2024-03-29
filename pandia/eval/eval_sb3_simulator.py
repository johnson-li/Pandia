import os
import gymnasium
from matplotlib import pyplot as plt
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO, SAC
import torch
from pandia import RESULTS_PATH, SB3_LOG_PATH
from pandia.agent.action import Action
from pandia.agent.curriculum_level import CURRICULUM_LEVELS
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.env_simple_simulator import WebRTCSimpleSimulatorEnv
from pandia.agent.utils import deep_update
from pandia.constants import K, M
from pandia.log_analyzer_sender import analyze_stream
from pandia.train.train_sb3_simple_simulator import CustomPolicy


def model_path(zoo=False, model='ppo'):
    if zoo:
        path = os.path.expanduser('~/sb3_logs/ppo')
        ids = [p.split('_')[-1] for p in os.listdir(path)]
        ids = sorted(ids)
        path = os.path.join(path, f'WebRTCSimpleSimulatorEnv_{ids[-1]}')
        return os.path.join(path, 'best_model.zip')
    else:
        path = os.path.expanduser('~/sb3_logs')
        models = [int(d[6:]) for d in os.listdir(path) if d.startswith('model_')]
        model_id = max(models) 
        path = os.path.join(path, f'model_{model_id}')
        print(f'Loading model from {path}')
        return os.path.join(path, 'best_model')


def main():
    # path = model_path()
    path = '/Users/johnson/sb3_logs/ppo/WebRTCSimpleSimulatorEnv_37/best_model'
    bw = 85 * M

    config = ENV_CONFIG
    deep_update(config, CURRICULUM_LEVELS[4])
    config['network_setting']['bandwidth'] = bw
    config['network_setting']['delay'] = .008
    config['gym_setting']['print_step'] = True
    config['gym_setting']['duration'] = 50
    config['gym_setting']['skip_slow_start'] = 0
    env = WebRTCSimpleSimulatorEnv(config=config, curriculum_level=None) # type: ignore

    # model = PPO.load(os.path.expanduser('~/sb3_logs/WebRTCSimpleSimulatorEnv_17600000_steps'), env)
    # model = RecurrentPPO.load(model_path(zoo=False), env)
    print(f'Loading model from {path}')
    model = PPO.load(path, env, custom_objects={'policy_class': CustomPolicy})
    # model = SAC.load(model_path(zoo=False), env)
    data = []
    obs, _ = env.reset()
    print(f'Eval with bw: {env.net_sample["bw"] / M:.02f} Mbps')
    rewards = []
    delays = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        obs_obj = env.observation
        delays.append(obs_obj.get_data(obs_obj.data[0][0], 'frame_decoded_delay', True))
        rewards.append(reward)
        if terminated or truncated:
            break
    actions = [a.bitrate / M for a in env.actions]
    bw = env.net_sample['bw']
    data.append((bw, np.mean(actions), np.mean(delays), np.mean(rewards)))
    print(f'bw: {bw / M:.02f}Mbps, bitrate: {np.mean(actions):.02f}Mbps, '
            f'delay: {np.mean(delays) * 1000:.02f}ms, reward: {np.mean(rewards):.02f}')
    env.close()

    # Plot evaluation results
    x = np.arange(len(actions)) 
    plt.close()
    fig, ax1 = plt.subplots()
    ax1.plot(x, actions, '.r')
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
    ax1.plot(x, actions, '.r')
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


if __name__ == "__main__":
    main()
