import argparse
import os
import time
import numpy as np
from stable_baselines3 import PPO
from pandia import RESULTS_PATH, SB3_LOG_PATH
from pandia.agent.env_client import WebRTCEnv0
from pandia.agent.env_config import ENV_CONFIG
from pandia.log_analyzer_sender import analyze_stream


def model_path():
    path = os.path.join(SB3_LOG_PATH, 'ppo')
    dirs = [f for f in os.listdir(path) if f.startswith('pandia_')]
    dirs = sorted(dirs, key=lambda x: int(x.split('_')[-1]))
    path = os.path.join(path, dirs[-1])
    print(f'Loading model from {path}')
    return os.path.join(path, 'best_model.zip')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--duration', type=int, default=10)
    parser.add_argument('-s', '--step_duration', type=float, default=ENV_CONFIG['step_duration'])
    parser.add_argument('-f', '--fake', action='store_true')
    parser.add_argument('-b', '--bw', type=int, default=1024*1024)
    parser.add_argument('-y', '--delay', type=int, default=5)
    parser.add_argument('-l', '--loss', type=int, default=2)
    parser.add_argument('-p', '--fps', type=int, default=30)
    parser.add_argument('-w', '--working_dir', type=str, default=os.path.join(RESULTS_PATH, 'eval_sb3'))
    args = parser.parse_args()
    fake_action = args.fake
    working_dir = args.working_dir
    action_keys = ENV_CONFIG['action_keys'] if not fake_action else ['fake']
    env_config={'bw': args.bw, 'delay': args.delay, 'loss': args.loss,
                'fps': args.fps, 'width': 2160,
                'print_step': True,
                'action_keys': action_keys,
                'client_id': 18, 'duration': args.duration,
                'step_duration': args.step_duration,
                'working_dir': working_dir,
                }
    env = WebRTCEnv0(**env_config)
    model = PPO.load(model_path(), env)
    start_ts = None
    obs, _ = env.reset()
    rewards = []
    while start_ts is None or time.time() - start_ts < args.duration:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if start_ts is None:
            start_ts = time.time()
        rewards.append(reward)
        if terminated or truncated:
            break
    env.close()
    print(f'Average reward: {np.mean(rewards):.02f}')
    os.system(f'mv {working_dir}/{env.log_name("sender")} {working_dir}/eval_sender.log')
    os.system(f'scp mobix:/tmp/{env.log_name("receiver")} {working_dir}/eval_receiver.log > /dev/null')
    analyze_stream(env.context, working_dir)


if __name__ == "__main__":
    main()
