import gymnasium as gym

from stable_baselines3 import SAC
from pandia.agent.env import WebRTCEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv


num_envs = 3
env_config = {'legacy_api': True}
envs = [WebRTCEnv(config=dict(env_config, port=7001+i)) for i in range(num_envs)]
env = SubprocVecEnv([lambda: envs[i] for i in range(num_envs)])
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    if done:
      obs = vec_env.reset()
