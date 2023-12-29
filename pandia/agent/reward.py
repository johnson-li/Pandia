from pandia.log_analyzer_sender import StreamingContext
from pandia.constants import M
import numpy as np


def reward(context: StreamingContext, net_sample, terminated=False, actions=list()):
    monitor_durations = list(sorted(context.monitor_blocks.keys()))
    mb = context.monitor_blocks[monitor_durations[0]]
    fps_score = 0
    # fps_score = - (1 - np.clip(mb.frame_fps / 30, 0, 1)) ** 2
    delay_score = 0
    for delay in [mb.frame_decoded_delay, mb.frame_egress_delay]:
        delay = max(0, delay) * 1000
        delay_score += - (delay / 100) ** 2
    quality_score = 3 * mb.frame_bitrate / net_sample['bw']
    res_score = 0
    # if penalty == 0:
    #     self.termination_ts = 0
    # if penalty > 0 and self.termination_ts == 0:
    #     # If unexpected situation lasts for 5s, terminate
    #     self.termination_ts = self.context.last_ts + self.termination_timeout
    # if mb.frame_fps < 1:
    #     penalty = 100
    stability_score = 0
    # if len(actions) >= 2:
    #     bitrates = [a.bitrate for a in actions[:-10]]
    #     stability_score = -np.std(bitrates) / M * 16
    score = res_score + quality_score + fps_score + delay_score + stability_score
    score = np.clip(score, -3, 3)
    return score