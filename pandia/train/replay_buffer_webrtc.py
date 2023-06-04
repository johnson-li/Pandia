import os
import pickle
import uuid
from pandia.agent.env import Action, WebRTCEnv
from stable_baselines3.common.buffers import ReplayBuffer


def save_replay_buffer(replay_buffer, path):
    with open(path, "wb+") as f:
        pickle.dump(replay_buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Replay buffer saved to {path}")


def limit_network(bw=1024, delay=10):
    if bw <= 0:
        os.system(f"ssh mobix '~/Workspace/Pandia/scripts/stop_traffic_control.sh'")
    else:
        os.system(f"ssh mobix '~/Workspace/Pandia/scripts/start_traffic_control.sh -d {delay} -b {bw}'")


def run(bw=1024, delay=10, width=240, name=str(uuid.uuid4())):
    print(f"Starting exp, bw: {bw}, delay: {delay}")
    limit_network(bw, delay)
    env = WebRTCEnv(config={
        'legacy_api': True,
        'enable_shm': False,
        'width': 144,
        'sender_log': 'sender_log.log',
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
    save_replay_buffer(replay_buffer, os.path.join(rb_dir, name))


def main():
    for width in [144, 240, 360, 720, 960, 1080]:
        for bw in range(100, 200, 100):
            for delay in [10, 20, 50, 100, 200, 500]:
                run(bw=bw, delay=delay, width=width, name=f'gcc_{width}p_{bw}kbps_{delay}ms.pkl')


if __name__ == "__main__":
    main()
