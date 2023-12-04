import os
from stable_baselines3 import PPO
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.env_emulator import WebRTCEmulatorEnv
from pandia.model.policies import CustomPolicy
from pandia.model.schedules import linear_schedule
from pandia.train.callbacks import SaveOnBestTrainingRewardCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor


def main():
    note = 'Train with emulator. Curricum level 0.'
    env_num = 8

    # model_pre = None
    model_pre = os.path.expanduser('~/sb3_logs/model_1/best_model')
    log_dir = os.path.expanduser('~/sb3_logs')
    models = [int(d[6:]) for d in os.listdir(log_dir) if d.startswith('model_')]
    if models:
        model_id = max(models) + 1
    else:
        model_id = 0
    log_dir = os.path.join(log_dir, f'model_{model_id}')
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'note.txt'), 'w') as f:
        f.write(note)
    config = ENV_CONFIG
    config['gym_setting']['print_step'] = True
    def make_env():
        env = WebRTCEmulatorEnv(config=config, curriculum_level=0)
        return env
    envs = SubprocVecEnv([make_env for _ in range(env_num)])
    envs = VecMonitor(envs, log_dir)
    checkpoint_callback = CheckpointCallback(save_freq=200_000, save_path=log_dir,
                                             name_prefix="WebRTCEmulatorEnv")
    best_model_callback = SaveOnBestTrainingRewardCallback(check_freq=2_000, log_dir=log_dir)
    if model_pre:
        model = PPO.load(model_pre, env=envs, verbose=1, custom_objects={'policy_class': CustomPolicy},
                         tensorboard_log=os.path.expanduser("~/sb3_tensorboard/WebRTCEmulatorEnv/"),
                         device="auto", batch_size=256, n_epochs=20, learning_rate=linear_schedule(0.00001))
    else:
        model = PPO(policy=CustomPolicy, env=envs, verbose=1, gamma=.8,
                    tensorboard_log=os.path.expanduser("~/sb3_tensorboard/WebRTCEmulatorEnv/"),
                    device="auto", batch_size=256, n_epochs=20, learning_rate=linear_schedule(0.0003))
    model.learn(total_timesteps=20_000_000, 
                callback=[checkpoint_callback, 
                          best_model_callback])


if __name__ == "__main__":
    main()
