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
from pandia.agent.observation import Observation  
from pandia.log_analyzer import main as analyzer_main
from pandia.log_analyzer_sender import analyze_stream


def run(bitrate=None, pacing_rate=None, working_dir=os.path.join(RESULTS_PATH, 'eval_rllib'), 
        duration=30, delay=5, loss=2, drl_path=None):
    if working_dir:
        Path(working_dir).mkdir(parents=True, exist_ok=True)
    bw = 10 * 1024
    action_keys = []
    if bitrate is not None:
        action_keys.append('bitrate')
    if pacing_rate is not None:
        action_keys.append('pacing_rate')
    if not action_keys:
        action_keys = ['fake']
    obs_keys = list(Observation.boundary().keys())
    env_config={'bw': bw, 'delay': delay, 'loss': loss,
                'fps': 30, 'width': 2160,
                'action_keys': action_keys, 'obs_keys': obs_keys,
                'client_id': 1, 'duration': duration,
                'step_duration': 1,
                'monitor_durations': [1, 2, 4],
                'working_dir': working_dir,
                }
    config = PPOConfig()\
        .rollouts(num_rollout_workers=0)\
        .environment(env='pandia', env_config=env_config)
    algo = config.build()

    if drl_path:
        algo.restore(os.path.expanduser(drl_path))
    env: WebRTCEnv0 = algo.workers.local_worker().env
    obs, info = env.reset()
    rewards = []
    for i in range(100000):
        if drl_path:
            action = algo.compute_single_action(obs, explore=False)
        else:
            action = Action(action_keys)
            if bitrate is not None:
                action.bitrate = bitrate
            if pacing_rate is not None:
                action.pacing_rate = pacing_rate
            act = action.array()
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
    args = parser.parse_args()
    # bitrate = 1024 * 50
    bitrate = None
    # pacing_rate = 1024 * 200
    pacing_rate = None
    working_dir = os.path.join(RESULTS_PATH, "eval_rllib")
    # path = '~/ray_results/PPO/PPO_None_97ccd_00000_0_2023-07-12_23-56-37/checkpoint_003600'
    path = None
    run(bitrate=bitrate, pacing_rate=pacing_rate, working_dir=working_dir, 
        duration=args.duration, delay=5, drl_path=path)


if __name__ == "__main__":
    main()