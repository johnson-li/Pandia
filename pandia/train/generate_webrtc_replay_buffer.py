import os
import pickle
import uuid
from pandia.agent.env import Action, WebRTCEnv
from stable_baselines3.common.buffers import ReplayBuffer


rb_dir = os.path.expanduser("~/Workspace/Pandia/resources/replay_buffer")
log_dir = os.path.expanduser("~/Workspace/Pandia/resources/logs")


def save_replay_buffer(replay_buffer, path):
    with open(path, "wb+") as f:
        pickle.dump(replay_buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Replay buffer saved to {path}")


def limit_network(bw=1024, delay=10):
    if bw <= 0:
        os.system(f"ssh mobix '~/Workspace/Pandia/scripts/stop_traffic_control.sh'")
    else:
        os.system(f"ssh mobix '~/Workspace/Pandia/scripts/start_traffic_control.sh -d {delay} -b {bw}'")


def run(bw=1024, delay=10, width=144, name=str(uuid.uuid4())):
    print(f"Starting exp, bw: {bw}, delay: {delay}")
    limit_network(bw, delay)
    env = WebRTCEnv(config={
        'legacy_api': True,
        'enable_shm': False,
        'width': width,
        'sender_log': os.path.join(log_dir, f'{name}.log'),
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
    if not os.path.exists(rb_dir):
        os.makedirs(rb_dir)
    save_replay_buffer(replay_buffer, os.path.join(rb_dir, f'{name}.pkl'))


def main():
    for width in [144, 240, 360, 720, 960, 1080]:
        for bw in [100, 200, 500, 800,
                   1 * 1024, 2 * 1024, 5 * 1024, 8 * 1024,
                   10 * 1024, 20 * 1024, 50 * 1024, 80 * 1024,
                   100 * 1024, 200 * 1024, 500 * 1024]:
            for delay in [10, 20, 50, 80, 100, 200, 500]:
                name = f'gcc_{width}p_{bw}kbps_{delay}ms'
                path = os.path.join(rb_dir, f'{name}.pkl')
                if not os.path.exists(path):
                    run(bw=bw, delay=delay, width=width, name=name)


if __name__ == "__main__":
    main()
