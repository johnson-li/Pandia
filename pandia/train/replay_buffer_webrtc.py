import os
import pickle
import uuid
from pandia.agent.env import Action, WebRTCEnv
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3 import SAC


def save_replay_buffer(replay_buffer, path):
    with open(path, "wb+") as f:
        pickle.dump(replay_buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Replay buffer saved to {path}")


def main():
    env = WebRTCEnv(config={
        'legacy_api': True,
        'enable_shm': False,
        'width': 720,
        })
    done = False
    action = Action()
    replay_buffer = ReplayBuffer(1024, env.observation_space, env.action_space)
    obs_pre = env.reset()
    for i in range(1000):
        obs, reward, done, info = env.step(action.array())
        action = env.get_action()
        print(f"GCC action: {action}")
        replay_buffer.add(obs_pre, obs, action.array(), reward, done, [{'TimeLimit.truncated': True}])
        obs_pre = obs
        if done:
            break
    env.close()
    rb_dir = os.path.expanduser("~/Workspace/Pandia/resources/replay_buffer")
    if not os.path.exists(rb_dir):
        os.makedirs(rb_dir)
    name = f"replay_buffer_gcc_{uuid.uuid4()}.pkl"
    save_replay_buffer(replay_buffer, os.path.join(rb_dir, name))


if __name__ == "__main__":
    main()
