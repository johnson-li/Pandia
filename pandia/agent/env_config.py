from pandia.agent.observation import Observation


ENV_CONFIG = {
    'action_keys' : ['bitrate'],
    'observation_keys' : Observation.boundary().keys(),
    'observation_durations' : [1, 2, 4],
}