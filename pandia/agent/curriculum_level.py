from pandia.agent.env_config import MIN_BW
from pandia.constants import K, M

MIN_NW_BW = 1 * M
MIN_BR = 200 * K

CURRICULUM_LEVELS = [
    {
        'network_setting': {
            'bandwidth': [5 * M, 5 * M],
            'delay': [0, .01],
        },
        'action_limit': {
            'bitrate': [MIN_BR, 10 * M],
        }
    },
    {
        'network_setting': {
            'bandwidth': [MIN_NW_BW, 20 * M],
            'delay': [0, .01],
        },
        'action_limit': {
            'bitrate': [MIN_BR, 20 * M],
        }
    },
    {
        'network_setting': {
            'bandwidth': [MIN_NW_BW, 50 * M],
            'delay': [0, .01],
        },
        'action_limit': {
            'bitrate': [MIN_BR, 50 * M],
        }
    },
    {
        'network_setting': {
            'bandwidth': [MIN_NW_BW, 100 * M],
            'delay': [0, .01],
        },
        'action_limit': {
            'bitrate': [MIN_BR, 100 * M],
        }
    },
    {
        'network_setting': {
            'bandwidth': [MIN_NW_BW, 200 * M],
            'delay': [0, .01],
        },
        'action_limit': {
            'bitrate': [MIN_BR, 200 * M],
        }
    },
]