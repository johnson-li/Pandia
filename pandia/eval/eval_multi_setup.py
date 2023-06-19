import os
import numpy as np
from ray.rllib.algorithms.sac import SACConfig
from pandia import DIAGRAMS_PATH
from pandia.agent.env import Action
from pandia.agent.env_client import WebRTCEnv0
from ray import tune

from pandia.log_analyzer import analyze_stream


def run(bitrate=1024, fps=10, width=720, working_dir=os.path.join(DIAGRAMS_PATH, 'eval_default')):
    enable_shm = True
    bw = 1024 * 1024
    tune.register_env('pandia', lambda config: WebRTCEnv0(**config))
    env_config={'enable_shm': enable_shm, 'width': 1080, 'bw': bw,
                            'client_id': 18, 'duration': 30, 'fps': fps,
                            'sender_log': '/tmp/eval_sender_log.txt',
                            'receiver_log': '/tmp/eval_receiver_log.txt'}
    config = SACConfig()\
        .rollouts(num_rollout_workers=0)\
        .environment(env='pandia', env_config=env_config)
    algo = config.build()
    # path = '~/ray_results/SAC/SAC_None_d24ac_00000_0_2023-06-10_15-03-33/checkpoint_000200'
    # algo.restore(os.path.expanduser(path))
    env: WebRTCEnv0 = algo.workers.local_worker().env
    obs, info = env.reset()
    rewards = []
    for i in range(1000):
        action = Action()
        action.bitrate[0] = bitrate
        action.resolution[0] = width
        obs, reward, done, truncated, info = env.step(action.array())
        rewards.append(reward)
        if done or truncated:
            break
    env.close()
    print(f'Average reward: {np.mean(rewards):.02f}')
    os.system('scp mobix:/tmp/eval_receiver_log.txt /tmp')
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    os.system(f'cp /tmp/eval_*.txt {working_dir}')
    analyze_stream(env.context, working_dir)


if __name__ == "__main__":
    for width in [360, 720, 1080]:
        for fps in [5, 10, 30]:
            for bitrate in [512, 1024, 2048, 4096]:
                prefix = f'eval_{width}p_{fps}_{bitrate}kbps'
                sender_path = os.path.join(DIAGRAMS_PATH, prefix, 'eval_sender_log.txt')
                if os.path.exists(sender_path) and os.path.getsize(sender_path) > 1024 * 1024:
                    print(f'Skipping {prefix}')
                    continue
                run(bitrate=bitrate, fps=fps, width=width, working_dir=os.path.join(DIAGRAMS_PATH, 'eval_multi_setup', prefix))
