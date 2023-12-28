from pandia.agent.env_config import MIN_BW
from pandia.constants import K, M


CURRICULUM_LEVELS = [
    {
        'network_setting': {
            'bandwidth': [1 * M, 10 * M],
            'delay': [0, .01],
        },
        'action_limit': {
            'bitrate': [1 * M, 10 * M],
        }
    },
    {
        'network_setting': {
            'bandwidth': [MIN_BW, 20 * M],
            'delay': [0, .01],
        },
        'action_limit': {
            'bitrate': [MIN_BW, 20 * M],
        }
    },
    {
        'network_setting': {
            'bandwidth': [MIN_BW, 50 * M],
            'delay': [0, .01],
        },
        'action_limit': {
            'bitrate': [MIN_BW, 50 * M],
        }
    },
    {
        'network_setting': {
            'bandwidth': [MIN_BW, 100 * M],
            'delay': [0, .01],
        },
        'action_limit': {
            'bitrate': [MIN_BW, 100 * M],
        }
    },
    {
        'network_setting': {
            'bandwidth': [MIN_BW, 200 * M],
            'delay': [0, .01],
        },
        'action_limit': {
            'bitrate': [MIN_BW, 200 * M],
        }
    },
]