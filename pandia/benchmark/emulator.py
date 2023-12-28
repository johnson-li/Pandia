import os

import numpy as np
from pandia import RESULTS_PATH
from pandia.agent.action import Action
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.env_emulator import WebRTCEmulatorEnv
from pandia.agent.observation import Observation
from pandia.analysis.stream_illustrator import generate_diagrams
from pandia.constants import M


def run(env, br, config, bw):
    action = Action(config['action_keys'])
    action.bitrate = br
    rewards = []
    frame_bitrates = []
    env.reset()
    while True:
        obs, reward, terminated, truncated, _ = env.step(action.array())
        obs = Observation.from_array(obs)
        rewards.append(reward)
        frame_bitrates.append(obs.get_data(obs.data[0][0], 'frame_bitrate', numeric=True))
        if terminated or truncated:
            break
    print(f'===BW: {env.net_sample["bw"] / M:.02f} mbps, bitrate: {br / M:.02f} mbps, '
          f'reward: {np.mean(rewards):.02f}, '
          f'frame bitrate: {np.mean(frame_bitrates) / M:.02f} mbps===')

def main():
    bw = 10 * M
    config = ENV_CONFIG
    config['network_setting']['bandwidth'] = bw
    config['network_setting']['delay'] = .001
    config['gym_setting']['print_step'] = True
    config['gym_setting']['print_period'] = 1
    config['gym_setting']['duration'] = 20
    config['gym_setting']['skip_slow_start'] = 0
    config['gym_setting']['step_duration'] = .1
    # config['gym_setting']['logging_path'] = '/tmp/pandia.log'
    config['gym_setting']['sb3_logging_path'] = '/tmp/pandia.log'
    env = WebRTCEmulatorEnv(config=config, curriculum_level=None) # type: ignore
    try:
        # for br in range(1 * M, 15 * M, M):
        br = 1 * M
        run(env, br, config, bw)
    except Exception as e:
        pass
    finally:
        env.close()
    
    path = os.path.join(RESULTS_PATH, "benchmark_emulator")
    generate_diagrams(path, env.context)


if __name__ == "__main__":
    main()
