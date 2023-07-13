import os
from pathlib import Path
from pandia import DIAGRAMS_PATH, RESULTS_PATH
from pandia.agent.env import Action
from pandia.agent.env_client import WebRTCEnv0
from ray import tune
import numpy as np
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.ppo import PPOConfig  
from pandia.log_analyzer import main as analyzer_main


def run(bitrate=0, pacing_rate=0, working_dir=os.path.join(RESULTS_PATH, 'eval_rllib'), 
        duration=30, delay=0, drl_path=None):
    Path(working_dir).mkdir(parents=True, exist_ok=True)
    bw = 1024 * 1024
    tune.register_env('pandia', lambda config: WebRTCEnv0(**config))
    env_config={'bw': bw, 'delay': delay, 'no_action': True,
                'client_id': 18, 'duration': duration,
                'working_dir': working_dir,}
    config = PPOConfig()\
        .rollouts(num_rollout_workers=0)\
        .environment(env='pandia', env_config=env_config)
    algo = config.build()

    if drl_path:
        algo.restore(os.path.expanduser(drl_path))
    env: WebRTCEnv0 = algo.workers.local_worker().env
    obs, info = env.reset()
    rewards = []
    for i in range(1000):
        if drl_path:
            action = algo.compute_single_action(obs, explore=False)
        else:
            action = Action()
            action.bitrate[0] = bitrate
            action.pacing_rate[0] = pacing_rate
            action = action.array()
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        if done or truncated:
            break
    env.close()
    print(f'Average reward: {np.mean(rewards):.02f}')
    os.system(f'mv {working_dir}/{env.log_name("sender")} {working_dir}/eval_sender.log')
    os.system(f'scp mobix:/tmp/{env.log_name("receiver")} {working_dir}/eval_receiver.log > /dev/null')
    analyzer_main(working_dir)

def main():
    bitrate = 1024 * 5
    pacing_rate = 1024 * 200
    working_dir = os.path.join(RESULTS_PATH, "eval_rllib")
    # path = '~/ray_results/PPO/PPO_None_97ccd_00000_0_2023-07-12_23-56-37/checkpoint_003600'
    path = None
    run(bitrate=bitrate, pacing_rate=pacing_rate, working_dir=working_dir, 
        duration=15, delay=5, drl_path=path)


if __name__ == "__main__":
    main()