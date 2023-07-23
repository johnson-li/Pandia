ENV_CONFIG = {
    'action_keys' : list(sorted(['bitrate'])),
    'observation_keys' : list(sorted([
        # 'frame_encoding_delay', 'frame_egress_delay', 'frame_recv_delay', 
        # 'frame_decoding_delay', 
        'frame_decoded_delay', 
        # 'frame_fps', 'frame_encoded_height',
        'frame_bitrate', 
        'bitrate',
        # 'pkt_ack_rate', 'pkt_loss_rate', 'pkt_delay_interval'
    ])),
    'observation_durations' : [1, 2, 3],
    'history_size' : 1,
    'bandwidth_range' : [1024, 3 * 1024],  # in kbps
    'bitrate_range': [300, 3 * 1024],  # in kbps
    'delay_range': [10, 10],  # in ms
    'loss_range': [0, 0],
    'step_duration': .1, # in s
    'duration': 30, # in s
    'width': 2160, # in px
    'fps': 30,
    'termination_timeout': 1, # in s
}