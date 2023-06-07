import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import pickle
from pandia.agent.env import WebRTCEnv
from stable_baselines3 import SAC
from stable_baselines3.common.save_util import open_path
from stable_baselines3.common.buffers import ReplayBuffer


def append_rb(rb: ReplayBuffer, rb_all: ReplayBuffer):
    size = rb.size()
    rb_all.observations[rb_all.pos:rb_all.pos + size] = rb.observations[:size].copy()
    rb_all.next_observations[rb_all.pos:rb_all.pos + size] = rb.next_observations[:size].copy()
    rb_all.actions[rb_all.pos:rb_all.pos + size] = rb.actions[:size].copy()
    rb_all.rewards[rb_all.pos:rb_all.pos + size] = rb.rewards[:size].copy()
    rb_all.dones[rb_all.pos:rb_all.pos + size] = rb.dones[:size].copy()
    rb_all.timeouts[rb_all.pos:rb_all.pos + size] = rb.timeouts[:size].copy()
    rb_all.pos += size


def main():
    env = WebRTCEnv()
    rb_dir = os.path.expanduser('~/Workspace/Pandia/resources/replay_buffer')
    rb_all = ReplayBuffer(100_000, env.observation_space, env.action_space)
    model = SAC('MlpPolicy', env)
    model.replay_buffer = rb_all
    for f in os.listdir(rb_dir):
        f = os.path.join(rb_dir, f)
        with open_path(f, "r", suffix="pkl") as file_handler:
            rb = pickle.load(file_handler)
        if rb.pos > 10:
            append_rb(rb, rb_all)
    print(f'Size of replay buffer: {rb_all.size()}')
    total_timesteps = 10_000
    callback = None
    total_timesteps, callback = model._setup_learn(
        total_timesteps,
        callback,
        True,
        'SAC',
        False,
    )
    for i in range(total_timesteps):
        print(f'Learning step {i}')
        model.train(1)
    model.save('gcc_offline')


if __name__ == '__main__':
    main()
