# All bitrates are in bps, all delays are in s, 
# all loss rates are in %, all data size are in bytes

from pandia.constants import M

RESOLUTION_LIST = [144, 240, 360, 480, 720, 960, 1080, 1440, 2160]

NORMALIZATION_RANGE = {
    'bandwidth': [1 * M, 100 * M],
    'bitrate': [1 * M, 100 * M],
    'fec_rate': [0, 255],
    'fps': [1, 60],
    'delay': [0, 1],
    'qp': [0, 255],
    'frame_size': [0, 100 * M],
    'loss_rate': [0, 1],
}

BOUNDARY = {
    # For action
    'fake': [0, 1], # placeholder
    'bitrate': NORMALIZATION_RANGE['bitrate'],
    'fps': NORMALIZATION_RANGE['fps'],
    # Limited by the software implementation,
    # The maximum egress rate of WebRTC is around 200 Mbps
    'pacing_rate': NORMALIZATION_RANGE['bandwidth'],
    'padding_rate': NORMALIZATION_RANGE['bandwidth'],  
    'fec_rate_key': NORMALIZATION_RANGE['fec_rate'],
    'fec_rate_delta': NORMALIZATION_RANGE['fec_rate'],
    'resolution': RESOLUTION_LIST,

    # For observation
    'frame_encoding_delay': NORMALIZATION_RANGE['delay'],
    'frame_egress_delay': NORMALIZATION_RANGE['delay'], 
    'frame_recv_delay': NORMALIZATION_RANGE['delay'], 
    'frame_decoding_delay': NORMALIZATION_RANGE['delay'], 
    'frame_decoded_delay': NORMALIZATION_RANGE['delay'], 
    'frame_fps': NORMALIZATION_RANGE['fps'],
    'frame_fps_decoded': NORMALIZATION_RANGE['fps'],
    'frame_qp': NORMALIZATION_RANGE['qp'],
    'frame_height': RESOLUTION_LIST,
    'frame_encoded_height': RESOLUTION_LIST, 
    'frame_size': NORMALIZATION_RANGE['frame_size'], 
    'frame_bitrate': NORMALIZATION_RANGE['bandwidth'],
    'frame_key_count': [0, 10], 
    'pkt_egress_rate': NORMALIZATION_RANGE['bandwidth'],
    'pkt_ack_rate': NORMALIZATION_RANGE['bandwidth'],
    'pkt_loss_rate': NORMALIZATION_RANGE['loss_rate'], 
    'pkt_trans_delay': NORMALIZATION_RANGE['delay'],
    'pkt_delay_interval': [0, 10], 
    'action_gap': [- 100 * M, 100 * M],
}

NETWORK_SETTING = {
    'bandwidth': [1 * M, 100 * M],
    'delay': [0, .1],
    'loss': [0, 0],
}

TRAINING_SETTING = {
    'bitrate_range': [1 * M, 10 * M],
}

VIDEO_SOURCE = {
    'resolution': 2160, 
    'fps': 30,
}

GYM_SETTING = {
    'step_duration': .01,
    'startup_delay': .1,
    'duration': 20,
    'observation_durations' : [1],
    'history_size' : 5,
    'print_step': False,
    'print_period': 1,
}

ENV_CONFIG = {
    'action_keys' : list(sorted(['bitrate'])),
    'observation_keys' : list(sorted([
        'frame_encoding_delay', 
        # 'frame_egress_delay', 'frame_recv_delay', 
        # 'frame_decoding_delay', 
        'frame_decoded_delay', 
        # 'frame_fps', 'frame_fps_decoded', 'frame_qp',
        # 'frame_height', 'frame_encoded_height', 'frame_size',
        'frame_bitrate', 
        # 'frame_key_count',
        'bitrate',
        'pkt_egress_rate', 'pkt_ack_rate', 
        'pkt_trans_delay', 'pkt_delay_interval',
        # 'pkt_loss_rate', 
        # 'pacing_rate', 'action_gap',
    ])),
    'action_static_settings' : {
        'bitrate': 1 * M,
        'pacing_rate': 200 * M, 
        'fps': 30,
        'fec_rate_key': 0,
        'fec_rate_delta': 0,
        'padding_rate': 0,
        'resolution': 1080,
    },
    'action_invalid_value': {
        'bitrate': 0,
        'pacing_rate': 0,
        'fps': 0,
        'fec_rate_key': 256,
        'fec_rate_delta': 256,
        'padding_rate': 0,  # TODO: 0 is still valid 
        'resolution': 0,
    },
    'boundary': BOUNDARY,
    'network_setting': NETWORK_SETTING,
    'training_setting': TRAINING_SETTING,
    'video_source': VIDEO_SOURCE,
    'gym_setting': GYM_SETTING,
}
