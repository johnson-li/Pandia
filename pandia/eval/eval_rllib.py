import os
import numpy as np
from ray.rllib.algorithms.sac import SACConfig
from pandia.agent.env import Action
from pandia.agent.env_client import WebRTCEnv0
from ray import tune

from pandia.log_analyzer import analyze_stream


def main():
    enable_shm = False
    bw = 1024
    tune.register_env('pandia', lambda config: WebRTCEnv0(**config))
    env_config={'enable_shm': enable_shm, 'width': 720, 'bw': bw,
                            'client_id': 18, 'duration': 30,
                            'sender_log': '/tmp/eval_sender_log.txt',
                            'receiver_log': '/tmp/eval_receiver_log.txt'}
    config = SACConfig()\
        .rollouts(num_rollout_workers=0)\
        .environment(env='pandia', env_config=env_config)
    algo = config.build()
    # path = '~/ray_results/SAC/SAC_None_d24ac_00000_0_2023-06-10_15-03-33/checkpoint_000200'
    # algo.restore(os.path.expanduser(path))
    env: WebRTCEnv0 = algo.workers.local_worker().env
    obs, info = env.reset()
    rewards = []
    for i in range(1000):
        action = Action()
        obs, reward, done, truncated, info = env.step(action.array())
        rewards.append(reward)
        if done or truncated:
            break
    env.close()
    print(f'Average reward: {np.mean(rewards)}')
    os.system('scp mobix:/tmp/eval_receiver_log.txt /tmp')
    analyze_stream(env.context)


if __name__ == "__main__":
    main()
