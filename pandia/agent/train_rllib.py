from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from pandia.agent.env import WebRTCEnv


algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment(env='WebRTCEnv-v0')
    .build()
)

for i in range(10):
    result = algo.train()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")
