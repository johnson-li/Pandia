import json
import os
from stable_baselines3 import PPO
from pandia.agent.curriculum_level import CURRICULUM_LEVELS
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.env_emulator import WebRTCEmulatorEnv
from pandia.model.policies import CustomPolicy
from pandia.model.schedules import linear_schedule
from pandia.train.callbacks import SaveOnBestTrainingRewardCallback, StartupCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor


def main():
    # model_pre = os.path.expanduser('~/sb3_logs/ppo/WebRTCSimpleSimulatorEnv_17/best_model')
    model_pre = None
    curriculum_level = 0
    algo = 'ppo'
    log_dir = os.path.expanduser(f'~/sb3_logs/{algo}')
    note = f'Train with variable bandwidth and delay. Curriculum level: {curriculum_level}. model_pre: {model_pre}'
    env_num = 8
    models = [int(d[18:]) for d in os.listdir(log_dir) if d.startswith('WebRTCEmulatorEnv_')]
    if models:
        model_id = max(models) + 1
    else:
        model_id = 0
    log_dir = os.path.join(log_dir, f'WebRTCEmulatorEnv_{model_id}')
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'note.txt'), 'w') as f:
        f.write(note)
    config = ENV_CONFIG
    config['gym_setting']['print_step'] = True
    config['gym_setting']['print_period'] = 1
    config['gym_setting']['duration'] = 10
    config['gym_setting']['skip_slow_start'] = 0
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump({'curriculum_level': CURRICULUM_LEVELS[curriculum_level] if curriculum_level is not None else None,
                   'config': config}, f)

    def make_env():
        env = WebRTCEmulatorEnv(config=config, curriculum_level=curriculum_level)
        return env
    envs = SubprocVecEnv([make_env for _ in range(env_num)])
    envs = VecMonitor(envs, log_dir)
    checkpoint_callback = CheckpointCallback(save_freq=200_000, save_path=log_dir,
                                             name_prefix="WebRTCEmulatorEnv")
    best_model_callback = SaveOnBestTrainingRewardCallback(check_freq=2_000, log_dir=log_dir)
    startup_callback = StartupCallback(log_dir=log_dir)
    if model_pre:
        model = PPO.load(model_pre, env=envs, verbose=1, custom_objects={'policy_class': CustomPolicy},
                         tensorboard_log=os.path.expanduser("~/sb3_tensorboard/WebRTCEmulatorEnv/"),
                         device="auto", batch_size=32, n_epochs=4, learning_rate=linear_schedule(0.00003))
    else:
        model = PPO(policy=CustomPolicy, env=envs, verbose=1, gamma=.8, n_steps=100,
                    tensorboard_log=os.path.expanduser("~/sb3_tensorboard/WebRTCEmulatorEnv/"),
                    device="auto", batch_size=32, n_epochs=4, learning_rate=linear_schedule(0.0003))
    model.learn(total_timesteps=20_000_000, 
                callback=[checkpoint_callback, startup_callback,
                          best_model_callback])


if __name__ == "__main__":
    main()
