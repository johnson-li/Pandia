from pandia.agent.observation import Observation


ENV_CONFIG = {
    'action_keys' : list(sorted(['bitrate'])),
    'observation_keys' : list(sorted(Observation.boundary().keys())),
    'observation_durations' : [1, 2, 4],
    'bandwidth_range' : [1024, 3 * 1024],  # in kbps
    'bitrate_range': [300, 3 * 1024],  # in kbps
    'delay_range': [0, 100],  # in ms
    'step_duration': .1, # in s
}