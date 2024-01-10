import argparse
import os
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from pandia import RESULTS_PATH
from pandia.agent.action import Action
from pandia.agent.env_emulator import WebRTCEmulatorEnv
from pandia.agent.curriculum_level import CURRICULUM_LEVELS
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.utils import deep_update
from pandia.analysis.stream_illustrator import generate_diagrams
from pandia.constants import M
from pandia.train.train_sb3_simple_simulator import CustomPolicy


def main(model_id=None, bw=7 * M):
    if model_id is None:
        log_dir = os.path.expanduser(f'~/sb3_logs/ppo')
        models = [int(d[18:]) for d in os.listdir(log_dir) if d.startswith('WebRTCEmulatorEnv_')]
        model_id = max(models)
    config = ENV_CONFIG
    deep_update(config, CURRICULUM_LEVELS[0])
    config['network_setting']['bandwidth'] = bw
    config['network_setting']['delay'] = .008
    config['gym_setting']['print_step'] = True
    config['gym_setting']['action_cap'] = False
    config['gym_setting']['print_period'] = 1
    config['gym_setting']['duration'] = 1000
    config['gym_setting']['skip_slow_start'] = 1
    env = WebRTCEmulatorEnv(config=config, curriculum_level=None) # type: ignore
    path = os.path.expanduser(f"~/sb3_logs/ppo/WebRTCEmulatorEnv_{model_id}/best_model")
    print(f'Loading model from {path}')
    model = PPO.load(path, env, custom_objects={'policy_class': CustomPolicy})
    obs, _ = env.reset()
    print(f'Eval with bw: {env.net_sample["bw"] / M:.02f} Mbps')
    rewards = []
    delays = []
    actions = []
    for i in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        obs_obj = env.observation
        act_obj = Action.from_array(action, env.action_keys)
        actions.append(act_obj.bitrate / M)
        delays.append(obs_obj.get_data(obs_obj.data[0][0], 'frame_decoded_delay', True))
        rewards.append(reward)
        if terminated or truncated:
            break
    bw = env.net_sample['bw']
    print(f'bw: {bw / M:.02f}Mbps, bitrate: {np.mean(actions):.02f}Mbps, '
            f'delay: {np.mean(delays) * 1000:.02f}ms, reward: {np.mean(rewards):.02f}')
    env.close()

    # Plot evaluation results
    output_dir = os.path.join(RESULTS_PATH, 'eval_sb3_emulator')
    os.makedirs(output_dir, exist_ok=True)

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
    plt.savefig(os.path.join(output_dir, 'bitrate_delay.pdf'))

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
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'bitrate_reward.pdf'))

    generate_diagrams(output_dir, env.context)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_id', type=int, default=None, help='model id')
    parser.add_argument('-b', '--bandwidth', type=float, default=7, help='bandwidth in mbps')
    args = parser.parse_args()
    main(model_id=args.model_id, bw=args.bandwidth * M)
