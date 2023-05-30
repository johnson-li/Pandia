import gymnasium as gym
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.logger import pretty_print
from pandia.agent.env import WebRTCEnv


def main():
    config = SACConfig()\
        .environment(env=WebRTCEnv)\
            .rollouts(num_rollout_workers=3)
    algo = config.build()

    for i in range(1000):
        result = algo.train()
        print(pretty_print(result))

        if i % 5 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")


if __name__ == "__main__":
    main()
