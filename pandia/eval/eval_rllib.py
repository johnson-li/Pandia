import argparse
import os
from pathlib import Path
from pandia import DIAGRAMS_PATH, RESULTS_PATH
from pandia.agent.action import Action
from pandia.agent.env_client import WebRTCEnv0
from ray import tune
import numpy as np
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.ppo import PPOConfig
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.observation import Observation  
from pandia.log_analyzer import main as analyzer_main
from pandia.log_analyzer_sender import analyze_stream


def run(drl_path, working_dir=os.path.join(RESULTS_PATH, 'eval_rllib'), 
        bw=1024 * 1024, duration=30, delay=5, loss=2, step_duration=.1,
        fake_action=False):
    if working_dir:
        Path(working_dir).mkdir(parents=True, exist_ok=True)
    action_keys = ENV_CONFIG['action_keys'] if not fake_action else ['fake']
    obs_keys = ENV_CONFIG['observation_keys']
    env_config={'bw': bw, 'delay': delay, 'loss': loss,
                'fps': 30, 'width': 2160,
                'print_step': True,
                'action_keys': action_keys, 'obs_keys': obs_keys,
                'client_id': 18, 'duration': duration,
                'step_duration': step_duration,
                'monitor_durations': [1, 2, 4],
                'working_dir': working_dir,
                }
    config = PPOConfig()\
        .rollouts(num_rollout_workers=0)\
        .environment(env='pandia', env_config=env_config)
    algo = config.build()
    if not fake_action:
        algo.restore(os.path.expanduser(drl_path))
    env: WebRTCEnv0 = algo.workers.local_worker().env
    obs, info = env.reset()
    rewards = []
    for i in range(100000):
        if fake_action:
            action = Action(action_keys)
            act = action.array()
        else:
            act = algo.compute_single_action(obs, explore=False)
            action = Action.from_array(act, action_keys)
        obs, reward, done, truncated, info = env.step(act)
        rewards.append(reward)
        if done or truncated:
            break
    env.close()
    print(f'Average reward: {np.mean(rewards):.02f}')
    os.system(f'mv {working_dir}/{env.log_name("sender")} {working_dir}/eval_sender.log')
    os.system(f'scp mobix:/tmp/{env.log_name("receiver")} {working_dir}/eval_receiver.log > /dev/null')
    # analyzer_main(working_dir)
    analyze_stream(env.context, working_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--duration', type=int, default=10)
    parser.add_argument('-s', '--step_duration', type=float, default=ENV_CONFIG['step_duration'])
    parser.add_argument('-f', '--fake', action='store_true')
    parser.add_argument('-b', '--bw', type=int, default=1024*1024)
    parser.add_argument('-y', '--delay', type=int, default=5)
    parser.add_argument('-l', '--loss', type=int, default=2)
    args = parser.parse_args()
    working_dir = os.path.join(RESULTS_PATH, "eval_rllib")
    path = '~/ray_results/PPO/PPO_None_705d5_00000_0_2023-07-20_07-03-52/checkpoint_000700'
    run(path, working_dir=working_dir, duration=args.duration,
        step_duration=args.step_duration, fake_action=args.fake,
        bw=args.bw, loss=args.loss, delay=args.delay)


if __name__ == "__main__":
    main()