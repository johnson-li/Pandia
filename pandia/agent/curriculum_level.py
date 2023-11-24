from pandia.constants import M


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
            'bandwidth': [1 * M, 20 * M],   
            'delay': [0, .01],
        },
        'action_limit': {
            'bitrate': [1 * M, 20 * M],
        }
    }, 
    {
        'network_setting': {
            'bandwidth': [1 * M, 50 * M],   
            'delay': [0, .01],
        },
        'action_limit': {
            'bitrate': [1 * M, 50 * M],
        }
    }, 
    {
        'network_setting': {
            'bandwidth': [1 * M, 100 * M],   
            'delay': [0, .01],
        },
        'action_limit': {
            'bitrate': [1 * M, 100 * M],
        }
    }, 
]