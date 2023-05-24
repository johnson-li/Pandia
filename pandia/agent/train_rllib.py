import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from pandia.agent.env import WebRTCEnv


def main():
    ray.init()
    config = PPOConfig().environment(env=WebRTCEnv)
    config.log_level = "DEBUG"
    algo = config.build()

    for i in range(10):
        result = algo.train()
        print(pretty_print(result))

        if i % 5 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")


if __name__ == "__main__":
    main()
