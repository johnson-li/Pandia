from gym import Space, spaces
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs


class WebRTCVecEnv(VecEnv):
    def __init__(self, num_envs: int, observation_space: Space, action_space: Space):
        super().__init__(num_envs, observation_space, action_space)

    def reset(self) -> VecEnvObs:
        return super().reset()
    
    def step_async(self, actions: np.ndarray) -> None:
        return super().step_async(actions)