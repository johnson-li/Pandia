import json
from typing import Any, Dict, List
import argparse
import numpy as np
import os
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import Schedule
from pandia import HYPERPARAMS_PATH
from pandia.agent.curriculum_level import CURRICULUM_LEVELS
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.env_simple_simulator import WebRTCSimpleSimulatorEnv
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.utils import ALGOS, StoreDict
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.results_plotter import ts2xy, plot_results
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.vec_env import VecMonitor
from pandia.agent.utils import deep_update
from pandia.constants import M
from pandia.model.policies import CustomPolicy
from pandia.model.schedules import linear_schedule
from pandia.train.callbacks import SaveOnBestTrainingRewardCallback, StartupCallback


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-tb", "--tensorboard-log", help="Tensorboard log dir", default=os.path.expanduser("~/sb3_tensorboard"), type=str)
    parser.add_argument("-i", "--trained-agent", help="Path to a pretrained agent to continue training", default="", type=str)
    parser.add_argument(
        "--truncate-last-trajectory",
        help="When using HER with online sampling the last trajectory "
        "in the replay buffer will be truncated after reloading the replay buffer.",
        default=True,
        type=bool,
    )
    parser.add_argument("-n", "--n-timesteps", help="Overwrite the number of timesteps", default=200_000_000, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--log-interval", help="Override log interval (default: -1, no change)", default=-1, type=int)
    parser.add_argument(
        "--eval-freq",
        help="Evaluate the agent every n steps (if negative, no evaluation). "
        "During hyperparameter optimization n-evaluations is used instead",
        default=5000,
        type=int,
    )
    parser.add_argument(
        "--optimization-log-path",
        help="Path to save the evaluation log and optimal policy for each hyperparameter tried during optimization. "
        "Disabled if no argument is passed.",
        type=str,
    )
    parser.add_argument("--eval-episodes", help="Number of episodes to use for evaluation", default=5, type=int)
    parser.add_argument("--n-eval-envs", help="Number of environments for evaluation", default=3, type=int)
    parser.add_argument("--save-freq", help="Save the model every n steps (if negative, no checkpoint)", default=20000, type=int)
    parser.add_argument(
        "--save-replay-buffer", help="Save the replay buffer too (when applicable)", action="store_true", default=False
    )
    parser.add_argument("-f", "--log-folder", help="Log folder", type=str, default=os.path.expanduser("~/sb3_logs"))
    parser.add_argument("--seed", help="Random generator seed", type=int, default=-1)
    parser.add_argument("--vec-env", help="VecEnv type", type=str, default="subproc", choices=["dummy", "subproc"])
    parser.add_argument("--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="cuda", type=str)
    parser.add_argument(
        "--n-trials",
        help="Number of trials for optimizing hyperparameters. "
        "This applies to each optimization runner, not the entire optimization process.",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--max-total-trials",
        help="Number of (potentially pruned) trials for optimizing hyperparameters. "
        "This applies to the entire optimization process and takes precedence over --n-trials if set.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-optimize", "--optimize-hyperparameters", action="store_true", default=False, help="Run hyperparameters search"
    )
    parser.add_argument(
        "--no-optim-plots", action="store_true", default=False, help="Disable hyperparameter optimization plots"
    )
    parser.add_argument("--n-jobs", help="Number of parallel jobs when optimizing hyperparameters", type=int, default=3)
    parser.add_argument(
        "--sampler",
        help="Sampler to use when optimizing hyperparameters",
        type=str,
        default="tpe",
        choices=["random", "tpe", "skopt"],
    )
    parser.add_argument(
        "--pruner",
        help="Pruner to use when optimizing hyperparameters",
        type=str,
        default="median",
        choices=["halving", "median", "none"],
    )
    parser.add_argument("--n-startup-trials", help="Number of trials before using optuna sampler", type=int, default=10)
    parser.add_argument(
        "--n-evaluations",
        help="Training policies are evaluated every n-timesteps // n-evaluations steps when doing hyperparameter optimization."
        "Default is 1 evaluation per 100k timesteps.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--storage", help="Database storage path if distributed optimization should be used", type=str, default=None
    )
    parser.add_argument("--study-name", help="Study name for distributed optimization", type=str, default=None)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environment package modules to import",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    )
    parser.add_argument(
        "-params",
        "--hyperparams",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)",
    )
    parser.add_argument(
        "-conf",
        "--conf-file",
        type=str,
        default=os.path.join(HYPERPARAMS_PATH, 'ppo.yml'),
        help="Custom yaml file or python package from which the hyperparameters will be loaded."
        "We expect that python packages contain a dictionary called 'hyperparams' which contains a key for each environment.",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        default=False,
        help="if toggled, this experiment will be tracked with Weights and Biases",
    )
    parser.add_argument("--wandb-project-name", type=str, default="sb3", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument(
        "-P",
        "--progress",
        action="store_true",
        default=False,
        help="if toggled, display a progress bar using tqdm and rich",
    )
    parser.add_argument(
        "-tags", "--wandb-tags", type=str, default=[], nargs="+", help="Tags for wandb run, e.g.: -tags optimized pr-123"
    )

    return parser.parse_args()


def main_zoo():
    args = parse_args()
    if args.seed < 0:
        # Seed but with a random one
        args.seed = np.random.randint(2**32 - 1, dtype="int64").item()  # type: ignore[attr-defined]
    set_random_seed(args.seed)
    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)
    if args.trained_agent != "":
        assert args.trained_agent.endswith(".zip") and os.path.isfile(
            args.trained_agent
        ), "The trained_agent must be a valid path to a .zip file"

    print(f"Seed: {args.seed}")

    exp_manager = ExperimentManager(
        args,
        args.algo,
        "WebRTCSimpleSimulatorEnv",
        args.log_folder,
        args.tensorboard_log,
        args.n_timesteps,
        args.eval_freq,
        args.eval_episodes,
        args.save_freq,
        args.hyperparams,
        args.env_kwargs,
        args.trained_agent,
        args.optimize_hyperparameters,
        args.storage,
        args.study_name,
        args.n_trials,
        args.max_total_trials,
        args.n_jobs,
        args.sampler,
        args.pruner,
        args.optimization_log_path,
        n_startup_trials=args.n_startup_trials,
        n_evaluations=args.n_evaluations,
        truncate_last_trajectory=args.truncate_last_trajectory,
        seed=args.seed,
        log_interval=args.log_interval,
        save_replay_buffer=args.save_replay_buffer,
        verbose=args.verbose,
        vec_env_type=args.vec_env,
        n_eval_envs=args.n_eval_envs,
        no_optim_plots=args.no_optim_plots,
        device=args.device,
        config=args.conf_file,
        show_progress=args.progress,
    )
    results = exp_manager.setup_experiment()
    if results is not None:
        model, saved_hyperparams = results

        # Normal training
        if model is not None:
            exp_manager.learn(model)
            exp_manager.save_trained_model(model)
    else:
        exp_manager.hyperparameters_optimization()


def main():
    model_pre = os.path.expanduser("~/sb3_logs/ppo/WebRTCSimpleSimulatorEnv_36/best_model")
    # model_pre = None
    curriculum_level = 4
    algo = 'ppo'
    note = f'Train with variable bandwidth and delay. Curriculum level: {curriculum_level}. model_pre: {model_pre}'
    env_num = 12
    log_dir = os.path.expanduser(f'~/sb3_logs/{algo}')
    models = [int(d[25:]) for d in os.listdir(log_dir) if d.startswith('WebRTCSimpleSimulatorEnv_')]
    if models:
        model_id = max(models) + 1
    else:
        model_id = 0
    log_dir = os.path.join(log_dir, f'WebRTCSimpleSimulatorEnv_{model_id}')
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'note.txt'), 'w') as f:
        f.write(note)
    config = ENV_CONFIG
    config['gym_setting']['print_step'] = True
    config['gym_setting']['print_period'] = 10
    config['gym_setting']['duration'] = 100
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump({'curriculum_level': CURRICULUM_LEVELS[curriculum_level] if curriculum_level is not None else None,
                   'config': config}, f)

    def make_env():
        env = WebRTCSimpleSimulatorEnv(config=config, curriculum_level=curriculum_level)
        return env
    envs = SubprocVecEnv([make_env for _ in range(env_num)])
    envs = VecMonitor(envs, log_dir)
    checkpoint_callback = CheckpointCallback(save_freq=20_000, save_path=log_dir,
                                             name_prefix="WebRTCSimpleSimulatorEnv")
    best_model_callback = SaveOnBestTrainingRewardCallback(check_freq=2_000, log_dir=log_dir)
    startup_callback = StartupCallback(log_dir=log_dir)
    if model_pre:
        model = PPO.load(model_pre, env=envs, verbose=1,
                tensorboard_log=os.path.expanduser("~/sb3_tensorboard/WebRTCSimpleSimulatorEnv"),
                device="auto", batch_size=256, n_epochs=20, learning_rate=linear_schedule(0.00003))
    else:
        model = PPO(policy=CustomPolicy, env=envs, verbose=1, gamma=.8,
                    tensorboard_log=os.path.expanduser("~/sb3_tensorboard/WebRTCSimpleSimulatorEnv"),
                    device="auto", batch_size=256, n_epochs=20, learning_rate=linear_schedule(0.0003))
    model.learn(total_timesteps=20_000_000,
                callback=[checkpoint_callback, startup_callback, best_model_callback])
    with open(os.path.join(log_dir, 'tensorboard.txt'), 'w') as f:
        f.write(str(model.logger.dir))


if __name__ == "__main__":
    # main_zoo()
    main()
