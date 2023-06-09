import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import gymnasium as gym
from stable_baselines3 import SAC
from pandia.agent.env import WebRTCEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv


def make_env(env_id):
  def _init():
    env_config = {'legacy_api': True, 'port': 7001 + env_id}
    env = WebRTCEnv(config=env_config)
    return env
  return _init


def main():
  num_envs = 4
  env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
  # env = WebRTCEnv(config={'legacy_api': True})
  model = SAC("MlpPolicy", env, verbose=1)
  model.learn(total_timesteps=10_000)

  vec_env = model.get_env()
  obs = vec_env.reset()
  for i in range(1000):
      action, _states = model.predict(obs, deterministic=True)
      obs, reward, done, info = vec_env.step(action)
      if done:
        obs = vec_env.reset()


if __name__ == '__main__':
  main()