import os
from ray.rllib.algorithms.sac import SACConfig
from pandia.agent.env import WebRTCEnv


def main():
    config = SACConfig()\
        .rollouts(num_rollout_workers=0)\
        .environment(env=WebRTCEnv, env_config={'enable_shm': False, 'width': 720})
    algo = config.build()
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
