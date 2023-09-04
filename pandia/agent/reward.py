from pandia.log_analyzer_sender import StreamingContext


def reward(context: StreamingContext, terminated=False):
    monitor_durations = list(sorted(context.monitor_blocks.keys()))
    mb = context.monitor_blocks[monitor_durations[0]]
    penalty = 0
    # fps_score = mb.frame_fps / 30
    fps_score = 0
    delay_score = 0
    for delay in [mb.frame_decoded_delay, mb.frame_egress_delay]:
        delay *= 1000
        delay_score += - delay ** 2 / 100 ** 2
    quality_score = mb.frame_bitrate / 1024 / 1024
    # res_score = mb.frame_height / 2160
    res_score = 0
    # if penalty == 0:
    #     self.termination_ts = 0
    # if penalty > 0 and self.termination_ts == 0:
    #     # If unexpected situation lasts for 5s, terminate
    #     self.termination_ts = self.context.last_ts + self.termination_timeout
    # if mb.frame_fps < 1:
    #     penalty = 100
    score = res_score + quality_score + fps_score + delay_score
    score = max(-10, score)
    return score