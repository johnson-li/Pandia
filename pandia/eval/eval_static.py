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


def run(bitrate=None, pacing_rate=None, bw=1024*1024, working_dir=os.path.join(RESULTS_PATH, 'eval_static'), 
        duration=30, delay=5, loss=2):
    if working_dir:
        Path(working_dir).mkdir(parents=True, exist_ok=True)
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
                'print_step': True,
                'action_keys': action_keys, 'obs_keys': obs_keys,
                'client_id': 18, 'duration': duration,
                'step_duration': ENV_CONFIG['step_duration'],
                'monitor_durations': [1, 2, 4],
                'working_dir': working_dir,
                }
    config = PPOConfig()\
        .rollouts(num_rollout_workers=0)\
        .environment(env='pandia', env_config=env_config)
    algo = config.build()

    env: WebRTCEnv0 = algo.workers.local_worker().env
    obs, info = env.reset()
    rewards = []
    for i in range(100000):
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
    parser.add_argument('-b', '--bw', type=int, default=1024*1024)
    parser.add_argument('-y', '--delay', type=int, default=5)
    parser.add_argument('-l', '--loss', type=int, default=2)
    parser.add_argument('-p', '--pacing_rate', type=int, default=10*1024)
    parser.add_argument('-r', '--bitrate', type=int, default=2*1024)
    parser.add_argument('-f', '--fake', action='store_true')
    parser.add_argument('-w', '--working_dir', type=str, 
                        default=os.path.join(RESULTS_PATH, "eval_rllib"))
    args = parser.parse_args()
    bitrate = args.bitrate if not args.fake else None
    pacing_rate = args.pacing_rate if not args.fake else None
    run(bitrate=bitrate, pacing_rate=pacing_rate, working_dir=args.working_dir, 
        duration=args.duration, delay=args.delay, bw=args.bw, loss=args.loss)


if __name__ == "__main__":
    main()