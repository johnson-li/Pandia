import os
from ray.rllib.algorithms.sac import SACConfig
from pandia.agent.env import WebRTCEnv
from pandia.agent.env_client import Env


def main():
    env = WebRTCEnv(config={'enable_shm': True, 'width': 720, 'port': 7009})
    config = SACConfig()\
        .rollouts(num_rollout_workers=0)\
        .environment(env=env)
    algo = config.build()
    path = '~/ray_results/SAC/SAC_None_d24ac_00000_0_2023-06-10_15-03-33/checkpoint_000200'
    algo.restore(os.path.expanduser("~/Workspace/Pandia/results/checkpoint"))
    env = algo.workers.local_worker().env
    obs, info = env.reset()
    rewards = []
    for i in range(1000):
        action = algo.compute_single_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        if done or truncated:
            break
    env.close()
    print(f'Average reward: {sum(rewards) / len(rewards)}')


if __name__ == "__main__":
    main()
    exit(0)
