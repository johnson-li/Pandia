import time
import gymnasium
import numpy as np
from ray import tune
from pandia import BIN_PATH
from pandia.agent.action import Action
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.observation import Observation
from pandia.agent.reward import reward
from pandia.agent.utils import sample
from pandia.log_analyzer_sender import FrameContext, PacketContext, StreamingContext


MTU = 1400


class FrameInfo:
    def __init__(self, frame_id) -> None:
        self.frame_id = frame_id
        self.resolution = 0
        self.bitrate = 0
        self.fps = 0
        self.capture_ts = .0
        self.encoded_size = .0
        self.rtp_packet_count = 0
        self.rtp_id_base = 0


class StreamingSimulator:
    def __init__(self, fps=30, bitrate=1024 * 1024, resolution=1080,
                 startup_delay=.1) -> None:
        self.fps = fps
        self.bitrate = bitrate
        self.resolution = resolution
        self.startup_delay = startup_delay
        self.frame_id = 0
        self.rtp_id = 0

    def next_frame_ts(self):
        return self.startup_delay + self.frame_id / self.fps

    def encoding_delay(self):
        return .01

    def decoding_delay(self):
        return .001

    def encoded_size(self):
        return self.bitrate / 8 / self.fps

    def get_next_frame(self):
        frame = FrameInfo(self.frame_id)
        frame.resolution = self.resolution
        frame.bitrate = self.bitrate
        frame.fps = self.fps
        frame.rtp_id_base = self.rtp_id
        frame.capture_ts = self.next_frame_ts()
        frame.encoded_size = self.encoded_size()
        frame.rtp_packet_count = int(frame.encoded_size / MTU) + 1
        self.rtp_id += frame.rtp_packet_count
        self.frame_id += 1
        return frame


class NetowkrSimulator:
    def __init__(self, sample, buffer_size=.1) -> None:
        self.rtt = sample['delay'] * 2
        self.bw = sample['bw'] * 1024
        self.buffer_size = buffer_size
        self.queue_delay = .0


class WebRTCSimpleSimulatorEnv(gymnasium.Env):
    def __init__(self, client_id=None, rank=None, duration=ENV_CONFIG['duration'], # Exp settings
                 resolution=ENV_CONFIG['width'], fps=ENV_CONFIG['fps'], # Source settings
                 bw=ENV_CONFIG['bandwidth_range'],  # Network settings
                 delay=ENV_CONFIG['delay_range'], loss=ENV_CONFIG['loss_range'], # Network settings
                 action_keys=ENV_CONFIG['action_keys'], # Action settings
                 obs_keys=ENV_CONFIG['observation_keys'], # Observation settings
                 monitor_durations=ENV_CONFIG['observation_durations'], # Observation settings
                 print_step=True, print_period=.2, log_path=None,# Logging settings
                 step_duration=ENV_CONFIG['step_duration'], # RL settings
                 termination_timeout=ENV_CONFIG['termination_timeout'] # Exp settings
                 ) -> None:
        super().__init__()
        # Exp settings
        self.duration = duration
        # Source settings
        self.resolution = resolution
        self.fps = fps
        # Network settings
        self.bw0 = bw  # in kbps
        self.delay0 = delay  # in ms
        self.loss0 = loss  # in %
        # Logging settings
        self.print_step = print_step
        self.print_period = print_period
        # RL settings
        self.step_duration = step_duration
        self.init_timeout = 10
        self.hisory_size = 1
        # RL state
        self.step_count = 0
        self.context: StreamingContext
        self.obs_keys = list(sorted(obs_keys))
        self.monitor_durations = list(sorted(monitor_durations))
        self.observation: Observation = Observation(self.obs_keys, 
                                                    durations=self.monitor_durations,
                                                    history_size=self.hisory_size)
        # ENV state
        self.streaming_simulator: StreamingSimulator = StreamingSimulator()
        self.network_simulator: NetowkrSimulator = NetowkrSimulator(self.sample_net_params())
        self.termination_ts = 0
        self.action_keys = list(sorted(action_keys))
        self.action_space = Action(action_keys).action_space()
        self.observation_space = \
            Observation(self.obs_keys, self.monitor_durations, self.hisory_size)\
                .observation_space()

    def sample_net_params(self):
        return {'bw': sample(self.bw0), 
                'delay': sample(self.delay0), 
                'loss': sample(self.loss0)
                }

    def reset(self, seed=None, options=None):
        self.context = StreamingContext(monitor_durations=self.monitor_durations)
        self.step_count = 0
        self.last_print_ts = 0
        self.network_simulator = NetowkrSimulator(self.sample_net_params())
        self.streaming_simulator = StreamingSimulator()
        self.observation = Observation(self.obs_keys, self.monitor_durations, self.hisory_size)
        return self.observation.array(), {}

    def on_new_frame_added(self, fi: FrameInfo):
        frame_id = fi.frame_id
        # Notify frame captured
        frame = FrameContext(frame_id, fi.capture_ts)
        frame.captured_at_utc = fi.capture_ts
        self.context.last_captured_frame_id = frame_id
        self.context.frames[frame_id] = frame
        [mb.on_frame_added(frame, fi.capture_ts) for mb in self.context.monitor_blocks.values()]
        # Notify frame encoding
        encoding_ts = fi.capture_ts
        frame.bitrate = fi.bitrate
        frame.fps = fi.fps
        frame.width = fi.resolution // 9 * 16
        frame.height = fi.resolution
        frame.encoding_at = encoding_ts
        [mb.on_frame_encoding(frame, encoding_ts) for mb in self.context.monitor_blocks.values()]
        # Notify frame encoded
        encoded_ts = frame.encoding_at + .01
        frame.encoded_at = encoded_ts
        frame.encoded_shape = (frame.width, frame.height)
        frame.encoded_size = int(fi.encoded_size)
        frame.qp = 0
        frame.is_key_frame = frame_id == 0
        [mb.on_frame_encoded(frame, encoded_ts) for mb in self.context.monitor_blocks.values()]
        frame.sequence_range[0] = fi.rtp_id_base
        frame.sequence_range[1] = fi.rtp_id_base + fi.rtp_packet_count - 1
        # Notify packet added and sent
        for i in range(fi.rtp_packet_count):
            rtp_id = fi.rtp_id_base + i
            send_ts = encoded_ts
            packet = PacketContext(rtp_id)
            packet.sent_at = send_ts
            packet.sent_at_utc = send_ts
            packet.payload_type = -1
            packet.seq_num = rtp_id
            packet.packet_type = 'video'
            packet.frame_id = frame_id
            packet.first_packet_in_frame = i == 0
            packet.last_packet_in_frame = i == fi.rtp_packet_count - 1
            packet.allow_retrans = True
            packet.retrans_ref = None
            packet.size = MTU
            self.context.packets[rtp_id] = packet
            self.context.packet_id_map[packet.seq_num] = rtp_id
            [mb.on_packet_added(packet, send_ts) for mb in self.context.monitor_blocks.values()]
            frame.rtp_packets[rtp_id] = packet
            self.context.last_egress_packet_id = packet.rtp_id
            [mb.on_packet_sent(packet, self.context.frames.get(packet.frame_id, None), send_ts) for mb in self.context.monitor_blocks.values()]
        # Notify packet received
        assmebled_ts = 0

        self.network_simulator.queue_delay = min(self.network_simulator.queue_delay, self.network_simulator.buffer_size)
        self.network_simulator.queue_delay -= 1 / self.streaming_simulator.fps 
        self.network_simulator.queue_delay = max(0, self.network_simulator.queue_delay)
        for i in range(fi.rtp_packet_count):
            self.network_simulator.queue_delay += MTU * 8 / self.network_simulator.bw
            transmission_delay = self.network_simulator.rtt + self.network_simulator.queue_delay
            rtp_id = fi.rtp_id_base + i
            packet = self.context.packets[rtp_id]
            packet.received_at_utc = transmission_delay + packet.sent_at_utc
            assmebled_ts = packet.received_at_utc
            packet.received = True
            packet.acked_at = packet.sent_at + transmission_delay - self.network_simulator.rtt / 2
            self.context.last_acked_packet_id = \
                max(rtp_id, self.context.last_acked_packet_id)
            [mb.on_packet_acked(packet, packet.acked_at) for mb in self.context.monitor_blocks.values()]
        # Notify frame decoding
        decoding_delay = .001
        frame.assembled_at_utc = assmebled_ts
        frame.decoding_at = assmebled_ts
        frame.decoding_at_utc = assmebled_ts
        frame.decoded_at = assmebled_ts + decoding_delay
        frame.decoded_at_utc = frame.decoded_at
        frame.assembled0_at = assmebled_ts
        self.context.last_decoded_frame_id = \
                max(frame.frame_id, self.context.last_decoded_frame_id)
        [mb.on_frame_decoding_updated(frame, frame.decoded_at) for mb in self.context.monitor_blocks.values()]
        self.context.last_ts = frame.decoded_at


    def step(self, action: np.ndarray):
        self.context.reset_step_context()
        act = Action.from_array(action, self.action_keys)
        self.streaming_simulator.bitrate = act.bitrate * 1024
        terminal_ts = self.step_duration * (self.step_count + 1)
        next_new_frame_ts = self.streaming_simulator.next_frame_ts()
        while next_new_frame_ts + self.network_simulator.rtt / 2 < terminal_ts:
            [mb.on_pacing_rate_set(next_new_frame_ts, act.pacing_rate * 1024) for mb in self.context.monitor_blocks.values()]
            frame = self.streaming_simulator.get_next_frame()
            self.on_new_frame_added(frame)
            next_new_frame_ts = self.streaming_simulator.next_frame_ts() 

        self.observation.append(self.context.monitor_blocks, act)
        r = reward(self.context)

        if self.print_step:
            if time.time() - self.last_print_ts > self.print_period:
                self.last_print_ts = time.time()
                print(f'#{self.step_count} '
                      f'R.w.: {r:.02f}, Act.: {act}Obs.: {self.observation}')
        self.step_count += 1
        truncated = next_new_frame_ts > self.duration
        return self.observation.array(), r, False, truncated, {}


tune.register_env('WebRTCSimpleSimulatorEnv', lambda config: WebRTCSimpleSimulatorEnv(**config))
gymnasium.register('WebRTCSimpleSimulatorEnv', entry_point='pandia.agent.env_simple_simulator:WebRTCSimpleSimulatorEnv', nondeterministic=True)


def test():
    env = gymnasium.make("WebRTCSimpleSimulatorEnv", 
                         bw=3 * 1024, delay=0,)
    action = Action(ENV_CONFIG['action_keys'])
    action.bitrate = int(1.8 * 1024)
    action.pacing_rate = 1000 * 1024
    episodes = 1
    try:
        for _ in range(episodes):
            env.reset()
            while True:
                _, _, terminated, truncated, _ = env.step(action.array())
                if terminated or truncated:
                    break
    except KeyboardInterrupt:
        pass
    env.close()


if __name__ == '__main__':
    test()
