from pandia.agent.observation import Observation


ENV_CONFIG = {
    'action_keys' : list(sorted(['bitrate'])),
    'observation_keys' : list(sorted([
        'frame_encoding_delay', 'frame_egress_delay', 'frame_recv_delay', 
        'frame_decoding_delay', 'frame_decoded_delay', 
        'frame_fps', 'frame_encoded_height',
        'frame_bitrate', 
        'pkt_ack_rate', 'pkt_loss_rate', 'pkt_delay_interval'
    ])),
    'observation_durations' : [1, 2, 3, 4, 5],
    'bandwidth_range' : [1024, 10 * 1024],  # in kbps
    'bitrate_range': [300, 10 * 1024],  # in kbps
    'delay_range': [0, 100],  # in ms
    'step_duration': .01, # in s
    'duration': 30, # in s
    'width': 2160, # in px
    'fps': 30,
    'termination_timeout': 1, # in s
}