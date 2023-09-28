ENV_CONFIG = {
    'action_keys' : list(sorted(['bitrate'])),
    'action_static_settings' : {
        'bitrate': 1024,
        'pacing_rate': 102400, 
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
    'observation_keys' : list(sorted([
        'frame_encoding_delay', 
        'frame_egress_delay', 'frame_recv_delay', 
        'frame_decoding_delay', 
        'frame_decoded_delay', 
        'frame_fps', 'frame_qp',
        'frame_height', 'frame_encoded_height', 'frame_size',
        'frame_bitrate', 'frame_key_count',
        'bitrate',
        'pkt_egress_rate', 'pkt_trans_delay',
        'pkt_ack_rate', 'pkt_loss_rate', 'pkt_delay_interval',
        'pacing_rate', 'action_gap',
    ])),
    'observation_durations' : [1, 2, 4],
    'history_size' : 1,
    'bandwidth_range' : [1024, 3 * 1024],  # in kbps
    'bitrate_range': [300, 3 * 1024],  # in kbps
    'delay_range': [0, 0],  # in ms
    'loss_range': [0, 0],
    'step_duration': .01, # in s
    'duration': 30, # in s
    'width': 2160, # in px
    'fps': 30,
    'termination_timeout': 1, # in s
}