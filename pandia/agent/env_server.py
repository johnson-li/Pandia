import os
import ray
from pandia import MODELS_PATH
from pandia.agent.env import Observation, Action
from ray import air, tune
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.env.policy_server_input import PolicyServerInput
from ray.rllib.offline import IOContext, InputReader



SERVER_ADDRESS = "localhost"
SERVER_PORT = 9900
CHECKPOINT_FILE = os.path.join(MODELS_PATH, "env_server.out")
NUM_WORKERS = 5
TUNE = False

def main():
    ray.init()
    run='SAC'
    checkpoint_path = CHECKPOINT_FILE

    def _input(ioctx: IOContext) -> InputReader:
        print(f'Worker index: {ioctx.worker_index}')
        if ioctx.worker_index > 0 or ioctx.worker.num_workers == 0:
            return PolicyServerInput(
                ioctx,
                SERVER_ADDRESS,
                SERVER_PORT + ioctx.worker_index - (1 if ioctx.worker_index > 0 else 0),
            )
        else:
            return None
    
    config = (
        SACConfig()
        .environment(
            env=None,
            observation_space=Observation.observation_space(legacy_api=False),
            action_space=Action.action_space(legacy_api=False),
        )
        .offline_data(input_=_input)
        .rollouts(
            num_rollout_workers=NUM_WORKERS,
            enable_connectors=False,
        )
        .evaluation(off_policy_estimation_methods={})
        .debugging(log_level="INFO")
    )
    config.rl_module(_enable_rl_module_api=False)
    config.training(_enable_learner_api=False)
    config.update_from_dict(
            {
                "num_steps_sampled_before_learning_starts": 100,
                "min_sample_timesteps_per_iteration": 200,
                "n_step": 3,
                "rollout_fragment_length": 4,
                "train_batch_size": 8,
            }
        )
    stop = {
        "training_iteration": 200,
        "timesteps_total": 100000,
        "episode_reward_mean": 100,
    }
    if TUNE:
        tune.Tuner(
            run, param_space=config, run_config=air.RunConfig(stop=stop, verbose=2)
        ).fit()
    else:
        algo = config.build()
        if os.path.exists(checkpoint_path):
            algo.restore(checkpoint_path)
        for _ in range(stop['timesteps_total']):
            algo.train()
            checkpoint = algo.save()
            print(f'Checkpoint saved at {checkpoint}')
            with open(checkpoint_path, 'w') as f:
                f.write(checkpoint)
        algo.stop()

if __name__ == "__main__":
    main()
