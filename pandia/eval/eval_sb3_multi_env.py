import os
import gymnasium
from matplotlib import pyplot as plt
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO, SAC
from pandia import RESULTS_PATH, SB3_LOG_PATH
from pandia.agent.action import Action
from pandia.agent.curriculum_level import CURRICULUM_LEVELS
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.env_simple_simulator import WebRTCSimpleSimulatorEnv
from pandia.agent.utils import deep_update
from pandia.constants import M
from pandia.log_analyzer_sender import analyze_stream


def model_path(zoo=False):
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
    bw_list = [i * M for i in range(2, 11)]
    rewards = []

    for bw in bw_list:
        config = ENV_CONFIG
        deep_update(config, CURRICULUM_LEVELS[0])
        config['network_setting']['bandwidth'] = bw
        config['network_setting']['delay'] = .002
        config['gym_setting']['duration'] = 50
        env = WebRTCSimpleSimulatorEnv(config=config, curriculum_level=None) # type: ignore
        path = model_path()
        model = PPO.load(path, env)
        print(f'Eval with bw: {env.bw0 / M:.02f} Mbps')
        obs, _ = env.reset()
        rewards0 = []
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            obs_obj = env.observation
            act_obj = Action.from_array(action, env.action_keys)
            rewards0.append(reward)
            if terminated or truncated:
                break
        env.close()
        rewards.append(np.mean(rewards0))

    # Plot evaluation results
    plt.close()
    plt.plot([b / M for b in bw_list], rewards)
    plt.xlabel('Bandwidth (Mbps)')
    plt.ylabel('Reward')
    plt.tight_layout()
    output_dir = os.path.join(RESULTS_PATH, 'eval_sb3')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'multi_env_bw_reward.pdf'))


if __name__ == "__main__":
    main()
