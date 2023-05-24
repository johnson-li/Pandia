import gym

from stable_baselines3 import PPO
from pandia.agent.env import WebRTCEnv

env = gym.make("WebRTCEnv-v0")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    if done:
      obs = env.reset()

env.close()