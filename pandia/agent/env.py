import gymnasium
import numpy as np
from pandia.agent.action import Action
from pandia.agent.curriculum_level import CURRICULUM_LEVELS

from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.observation import Observation
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

    def reset(self, seed=None, options=None):
        np.random.seed(seed)
        self.context = StreamingContext(monitor_durations=self.monitor_durations)
        self.actions.clear()
        self.step_count = 0
        self.sample_net_params()
        self.observation = Observation(self.obs_keys, self.monitor_durations, self.hisory_size)
        return self.observation.array(), {}
