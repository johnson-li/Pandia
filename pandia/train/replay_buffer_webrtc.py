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


def run(bw_limit=1024, name=str(uuid.uuid4())):
    os.system('sudo tc qdisc del dev lo root 2> /dev/null')
    if bw_limit > 0:
        # os.system('sudo tc qdisc add dev lo root netem delay 50ms')
        os.system(f'sudo tc qdisc add dev lo root handle 1: htb default 10')
        os.system(f'sudo tc class add dev lo parent 1: classid 1:1 htb rate {bw_limit}kbit')
        os.system(f'sudo tc filter add dev lo parent 1: protocol ip prio 1 u32 match ip src 0.0.0.0/0 flowid 1:1')
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
    name = f"replay_buffer_gcc_{name}.pkl"
    save_replay_buffer(replay_buffer, os.path.join(rb_dir, name))
    os.system('sudo tc qdisc del dev lo root 2> /dev/null')


def main():
    for i in range(100, 200, 100):
        run(bw_limit=i, name=str(i))


if __name__ == "__main__":
    main()
