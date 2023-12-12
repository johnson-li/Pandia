import gymnasium
import numpy as np
from pandia.agent.action import Action
from pandia.agent.curriculum_level import CURRICULUM_LEVELS
from stable_baselines3.common.utils import set_random_seed
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.observation import Observation
from pandia.agent.reward import reward
from pandia.agent.utils import deep_update, sample
from pandia.log_analyzer_sender import StreamingContext


class WebRTCEnv(gymnasium.Env):
    def __init__(self, config=ENV_CONFIG, curriculum_level=0) -> None: 
        super().__init__()
        if curriculum_level is not None:
            deep_update(config, CURRICULUM_LEVELS[curriculum_level])
        # Exp settings
        self.duration = config['gym_setting']['duration']
        self.startup_delay = config['gym_setting']['startup_delay']
        self.skip_slow_start = config['gym_setting']['skip_slow_start']
        # Source settings
        self.resolution = config['video_source']['resolution']
        self.fps = config['video_source']['fps']
        # Network settings
        self.bw0 = config['network_setting']['bandwidth']
        self.delay0 = config['network_setting']['delay']
        self.loss0 = config['network_setting']['loss']
        self.buffer_size0 = .1
        self.queue_delay0 = .25
        self.net_sample: dict
        # Logging settings
        self.print_step = config['gym_setting']['print_step']
        self.print_period = config['gym_setting']['print_period']
        # RL settings
        self.step_duration = config['gym_setting']['step_duration']
        self.hisory_size = config['gym_setting']['history_size']
        # RL state
        self.step_count = 0
        self.context: StreamingContext
        self.obs_keys = list(sorted(config['observation_keys']))
        self.monitor_durations = list(sorted(config['gym_setting']['observation_durations']))
        self.observation: Observation = Observation(self.obs_keys, 
                                                    durations=self.monitor_durations,
                                                    history_size=self.hisory_size)
        self.actions = []
        # ENV state
        self.termination_ts = 0
        self.action_keys = list(sorted(ENV_CONFIG['action_keys']))
        self.action_space = Action(self.action_keys, boundary=config['boundary']).action_space()
        self.action_limit = config.get('action_limit', {})
        print(f'action limit: {self.action_limit}')
        self.observation_space = \
            Observation(self.obs_keys, self.monitor_durations, self.hisory_size)\
                .observation_space()

    def sample_net_params(self):
        self.net_sample = {'bw': sample(self.bw0),
                'delay': sample(self.delay0),
                'loss': sample(self.loss0),
                'buffer_size': sample(self.buffer_size0),
                'queue_delay': sample(self.queue_delay0)}
        return self.net_sample

    def step0(self):
        raise NotImplementedError()

    # def step(self, action: np.ndarray):
    #     limit = Action.action_limit(self.action_keys, limit=self.action_limit)
    #     action = np.clip(action, limit.low, limit.high)
    #     self.context.reset_step_context()
    #     act = Action.from_array(action, self.action_keys)
    #     self.actions.append(act)
    #     self.step0()

    #     self.observation.append(self.context.monitor_blocks, act)
    #     r = reward(self.context, self.net_sample, actions=self.actions)

    #     if self.print_step and (self.step_count * self.step_duration - self.last_print_ts) >= self.print_period:
    #         self.last_print_ts = self.step_count * self.step_duration
    #         print(f'#{self.step_count} [{self.step_count * self.step_duration:.02f}s] '
    #                 f'R.w.: {r:.02f}, Act.: {act}, Obs.: {self.observation}')
    #     self.step_count += 1
    #     truncated = next_new_frame_ts > self.duration + self.skip_slow_start
    #     return self.observation.array(), r, False, truncated, {}

    def reset(self, seed=None, options=None):
        if seed is None:
            seed = np.random.randint(0xffffff)
        set_random_seed(seed, True)
        self.context = StreamingContext(monitor_durations=self.monitor_durations)
        self.actions.clear()
        self.step_count = 0
        self.sample_net_params()
        self.observation = Observation(self.obs_keys, self.monitor_durations, self.hisory_size)
        return self.observation.array(), {}
