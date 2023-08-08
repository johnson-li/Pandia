import os

from pandia import RESULTS_PATH
from pandia.agent.env_client import WebRTCEnv0

from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.observation import Observation
from pandia.log_analyzer_sender import StreamingContext, parse_line


class PlaybackEnv():
    def __init__(self, path, step_duration=ENV_CONFIG['step_duration']) -> None:
        self.path = path
        self.sender_log = os.path.join(path, 'eval_sender.log')
        self.step_duration = step_duration
    
    def run(self):
        rewards = []
        context = StreamingContext(monitor_durations=ENV_CONFIG['observation_durations'])
        step = 0
        observation = Observation()
        with open(self.sender_log, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parse_line(line, context)
                    if context.start_ts > 0:
                        current_step = int((context.last_ts - context.start_ts) / self.step_duration)
                        if current_step > step:
                            reward = WebRTCEnv0.reward(context)
                            observation.append(context.monitor_blocks, None)
                            rewards.append(reward)
                            print(f'#{step} R.w.: {reward:.02f}, Obs.: {observation}')
                            print(observation.data)
                            step = current_step

def main():
    env = PlaybackEnv(os.path.join(RESULTS_PATH, "eval_rllib"), .1)
    env.run()


if __name__ == "__main__":
    main()
