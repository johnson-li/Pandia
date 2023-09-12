import os
import socket
from struct import unpack
import time
import uuid
import docker
import threading
import gymnasium
import numpy as np
from pandia import RESULTS_PATH
from pandia.agent.action import Action
from pandia.agent.env_client import ActionHistory
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.observation import Observation
from pandia.agent.reward import reward
from pandia.constants import WEBRTC_SENDER_SB3_PORT
from pandia.log_analyzer_sender import PACKET_TYPES, FrameContext, PacketContext, StreamingContext, analyze_stream
from ray import tune
from typing import Any, Optional
from docker.models.containers import Container
from pandia.log_analyzer_sender import kTimeWrapPeriod


NETWORK = None
STUN: Container
TS = time.time()


def log(msg):
    print(f'[{time.time() - TS:.02f}] {msg}', flush=True)


def setup_containers():
    global NETWORK
    global STUN
    client = docker.from_env()
    try:
        NETWORK = client.networks.get('sb3_net')
    except docker.errors.NotFound: # type: ignore
        NETWORK = client.networks.create('sb3_net', driver='bridge')
    try:
        STUN = client.containers.get('sb3_stun') # type: ignore
    except Exception: # type: ignore
        os.system(f'docker run -d --rm --network sb3_net --name sb3_stun ich777/stun-turn-server')
        STUN = client.containers.get('sb3_stun') # type: ignore


class ObservationThread(threading.Thread):
    def __init__(self, sock) -> None:
        super().__init__()
        self.sock: socket.socket = sock
        self.context: Optional[StreamingContext] = None
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()
        self.sock.close()

    def run(self) -> None:
        while not self.stop_event.is_set():
            data, addr = self.sock.recvfrom(1024)
            # print(f'Received {len(data)} bytes from {addr}', flush=True)
            msg_type = unpack('Q', data[:8])[0]
            data = data[8:]
            if msg_type == 0:
                log(f'WebRTC receiver is started.')
                continue
            try:
                self.parse_data(data, msg_type)
            except Exception as e:
                data_hex = ''.join('{:02x}'.format(x) for x in data)
                log(f'Msg type: {msg_type}, Error: {e}, Data: {len(data)} bytes')

    def parse_data(self, data, msg_type):
        if self.context is None:
            log(f'ERROR: context is not initialized yet.')
            return
        context: StreamingContext = self.context
        if msg_type == 1:  # Frame captured
            ts, frame_id, width, height, frame_ts, frame_utc_ts = unpack('QQQQQQ', data)
            ts /= 1000
            frame_utc_ts /= 1000
            frame = FrameContext(frame_id, ts)
            frame.captured_at_utc = frame_utc_ts
            context.last_captured_frame_id = frame_id
            context.frames[frame_id] = frame
            [mb.on_frame_added(frame, ts) for mb in context.monitor_blocks.values()]
            # log(f'Frame captured: {frame_id}')
        elif msg_type == 2:  # Apply FEC rates
            pass
        elif msg_type == 3:  # Setup codec
            ts, = unpack('Q', data)
            ts /= 1000
            if context.start_ts <= 0:
                context.start_ts = ts
                log(f'Codec is setup, start ts: {ts}')
        elif msg_type == 4:  # Packet added 
            ts, rtp_id, seq_num, first_in_frame, last_in_frame, frame_id, rtp_type, \
                retrans_seq_num, allow_retrans, size = unpack('QQQQQQQQQQ', data)
            ts /= 1000
            if rtp_id > 0:
                # log(f'Packet sent: {rtp_id}')
                packet = PacketContext(rtp_id)
                packet.seq_num = seq_num
                packet.packet_type = PACKET_TYPES[rtp_type]
                packet.frame_id = frame_id
                packet.first_packet_in_frame = first_in_frame
                packet.last_packet_in_frame = last_in_frame
                packet.allow_retrans = allow_retrans
                packet.retrans_ref = retrans_seq_num
                packet.size = size
                context.packets[rtp_id] = packet
                context.packet_id_map[seq_num] = rtp_id
                [mb.on_packet_added(packet, ts) for mb in context.monitor_blocks.values()]
                if rtp_type == 'rtx':
                    packet.frame_id = context.packets[context.packet_id_map[retrans_seq_num]].frame_id
                if packet.frame_id > 0 and frame_id in context.frames:
                    frame: FrameContext = context.frames[frame_id]
                    frame.rtp_packets[rtp_id] = packet
                    if rtp_type == 'rtx':
                        original_rtp_id = context.packet_id_map[retrans_seq_num]
                        frame.retrans_record.setdefault(original_rtp_id, []).append(rtp_id)
                    if rtp_type == 'video':
                        frame.sequence_range[0] = min(frame.sequence_range[0], seq_num)
                        frame.sequence_range[1] = max(frame.sequence_range[1], seq_num)
        elif msg_type == 5:  # Start encoding
            ts, frame_id, height, bitrate, key_frame, fps = unpack('QQQQQQ', data)
            ts /= 1000
            bitrate *= 1024
            context.action_context.resolution = height 
            if context.action_context.bitrate <= 0:
                context.action_context.bitrate = bitrate
            if frame_id in context.frames:
                frame: FrameContext = context.frames[frame_id]
                frame.bitrate = bitrate
                frame.fps = fps
                frame.height = height
                frame.encoding_at = ts
                [mb.on_frame_encoding(frame, ts) for mb in context.monitor_blocks.values()]
            # log(f'Frame encoding started: {frame_id}, ts: {ts}')
        elif msg_type == 6:  # Finish encoding
            ts, frame_id, height, frame_size, is_key, qp = unpack('QQQQQQ', data)
            ts /= 1000
            if frame_id in context.frames:
                frame = context.frames[frame_id]
                frame.encoded_at = ts
                frame.encoded_shape = (0, height)
                frame.encoded_size = frame_size
                frame.qp = qp
                frame.is_key_frame = is_key != 0
                [mb.on_frame_encoded(frame, ts) for mb in context.monitor_blocks.values()]
            else:
                log(f'ERROR: frame {frame_id} is not found, the last one is {context.last_captured_frame_id}')
        elif msg_type == 7:  # RTCP RTT 
            ts, rtt = unpack('QQ', data)
            ts /= 1000
            rtt /= 1000
            context.rtt_data.append((ts, rtt))
        elif msg_type == 8:  # Frame decoding ack 
            ts, rtp_sequence, received_ts_utc, decoding_ts_utc, decoded_ts_utc = unpack('QQQQQ', data)
            ts /= 1000
            received_ts_utc /= 1000
            decoding_ts_utc /= 1000
            decoded_ts_utc /= 1000
            rtp_id = context.packet_id_map.get(rtp_sequence, -1)
            if rtp_id > 0 and rtp_id in context.packets:
                frame_id = context.packets[rtp_id].frame_id
                if frame_id in context.frames:
                    frame: FrameContext = context.frames[frame_id]
                    frame.assembled_at_utc = received_ts_utc
                    frame.decoding_at_utc = decoding_ts_utc
                    frame.decoded_at_utc = decoded_ts_utc
                    frame.decoded_at = ts
                    frame.decoding_at = ts - (decoded_ts_utc - decoding_ts_utc) / 1000
                    frame.assembled0_at = ts - (decoded_ts_utc - received_ts_utc) / 1000
                    context.last_decoded_frame_id = \
                            max(frame.frame_id, context.last_decoded_frame_id)
                    [mb.on_frame_decoding_updated(frame, ts) for mb in context.monitor_blocks.values()]
                    # log(f'Frame decoded: {frame_id}')
        elif msg_type == 9:  # Send video
            pass
        elif msg_type == 10:  # Apply bitrate 
            ts, bitrate, pacing_rate = unpack('QQQ', data)
            ts /= 1000
            bitrate *= 1024
            pacing_rate *= 1024
            context.drl_bitrate.append((ts, bitrate))
            context.drl_pacing_rate.append((ts, pacing_rate))
        elif msg_type == 11:  # Apply pacing rate
            ts, pacing_rate, padding_rate = unpack('QQQ', data)
            ts /= 1000
            pacing_rate *= 1024
            padding_rate *= 1024
            context.action_context.pacing_rate = pacing_rate
            context.networking.pacing_rate_data.append([ts, pacing_rate, padding_rate])
            [mb.on_pacing_rate_set(ts, pacing_rate) for mb in context.monitor_blocks.values()]
        elif msg_type == 12:  # RTCP feedback
            ts, count = unpack('QQ', data[:16])
            ts /= 1000
            data = data[16:]
            size = 64
            seq_nums = unpack(f'{size}H', data[:size * 2])[:count]
            data = data[size * 2:]
            losts = unpack(f'{size}B', data[:size])[:count]
            data = data[size:]
            ts_list = unpack(f'{size}Q', data)[:count]
            
            pkt_pre = None
            for rtp_id, lost, received_at in zip(seq_nums, losts, ts_list):
                # The recv time is wrapped by kTimeWrapPeriod.
                # The fixed value 1570 should be calculated according to the current time.
                offset = int(time.time() * 1000 / kTimeWrapPeriod) - 1
                received_at = (int(received_at) + offset * kTimeWrapPeriod) / 1000
                rtp_id = int(rtp_id)
                if rtp_id in context.packets:
                    packet = context.packets[rtp_id]
                    packet.received_at_utc = received_at
                    packet.received = lost != 1
                    packet.acked_at = ts
                    context.last_acked_packet_id = \
                        max(rtp_id, context.last_acked_packet_id)
                    [mb.on_packet_acked(packet, pkt_pre, ts) for mb in context.monitor_blocks.values()]
                    if packet.received:
                        pkt_pre = packet
        elif msg_type == 13:  # Packet sent
            ts, rtp_id, payload_type, size, utc = unpack('QqQQQ', data)
            ts /= 1000
            utc /= 1000
            if rtp_id > 0:
                if rtp_id not in context.packets:
                    print(f'ERROR: packet {rtp_id} is not found, the last one is {context.last_egress_packet_id}')
                else:
                    packet: PacketContext = context.packets[rtp_id]
                    packet.payload_type = payload_type
                    packet.size = size
                    packet.sent_at = ts 
                    packet.sent_at_utc = utc
                    context.last_egress_packet_id = max(rtp_id, context.last_egress_packet_id)
                    [mb.on_packet_sent(packet, context.frames.get(packet.frame_id, None), ts) for mb in context.monitor_blocks.values()]
        else:
            log(f'Unknown message type: {data[0]}')


class WebRTContainerEnv(gymnasium.Env):
    def __init__(self, client_id=None, rank=None, duration=ENV_CONFIG['duration'], # Exp settings
                 width=ENV_CONFIG['width'], fps=ENV_CONFIG['fps'], # Source settings
                 bw=ENV_CONFIG['bandwidth_range'],  # Network settings
                 delay=ENV_CONFIG['delay_range'], loss=ENV_CONFIG['loss_range'], # Network settings
                 action_keys=ENV_CONFIG['action_keys'], # Action settings
                 obs_keys=ENV_CONFIG['observation_keys'], # Observation settings
                 monitor_durations=ENV_CONFIG['observation_durations'], # Observation settings
                 print_step=True, print_period=2,# Logging settings
                 step_duration=ENV_CONFIG['step_duration'], # RL settings
                 termination_timeout=ENV_CONFIG['termination_timeout'] # Exp settings
                 ) -> None:
        super().__init__()
        print(f'Creating WebRTCSb3Env with client_id={client_id}, rank={rank}')
        self.duration = duration
        self.termination_timeout = termination_timeout
        # Source settings
        self.width = width
        self.fps = fps
        # Network settings
        self.bw0 = bw  # in kbps
        self.delay0 = delay  # in ms
        self.loss0 = loss  # in %
        self.net_params = {}
        # Logging settings
        self.print_step = print_step
        self.print_period = print_period
        self.stop_event: Optional[threading.Event] = None
        self.logging_buf = []
        # RL settings
        self.step_duration = step_duration
        self.init_timeout = 30
        self.hisory_size = 1
        # RL state
        self.step_count = 0
        self.context: StreamingContext
        self.obs_keys = list(sorted(obs_keys))
        self.monitor_durations = list(sorted(monitor_durations))
        self.observation: Observation = Observation(self.obs_keys, durations=self.monitor_durations,
                                                    history_size=self.hisory_size)
        # ENV state
        self.receiver_container: Container
        self.sender_container: Container
        self.container_socket: socket.socket
        self.termination_ts = 0
        self.action_keys = list(sorted(action_keys))
        self.action_space = Action(action_keys).action_space()
        self.observation_space = \
            Observation(self.obs_keys, self.monitor_durations, self.hisory_size)\
                .observation_space()
        # Tracking
        self.start_ts = 0
        self.action_history = ActionHistory()
        self.docker_client = docker.from_env()
        self.obs_socket, self.obs_port = self.create_observer()
        self.obs_thread = ObservationThread(self.obs_socket)
        self.obs_thread.start()
        self.cid = ''
        self.start_containers()

    def create_observer(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        port = 0
        for i in range(100):
            try:
                port = 9990 - i
                sock.bind(('0.0.0.0', port))
                break
            except OSError as e:
                print(e)
        print(f'Listening on port {port}')
        return sock, port

    def start_containers(self):
        cid = str(uuid.uuid4())[:8]
        self.cid = cid
        print('cid: ', cid)
        cmd = f'docker run -d --rm --network sb3_net --name sb3_receiver_{cid} --hostname sb3_receiver_{cid} '\
              f'--runtime=nvidia --gpus all '\
              f'--cap-add=NET_ADMIN -e NVIDIA_DRIVER_CAPABILITIES=all '\
              f'-v /tmp/pandia:/tmp '\
              f'--env STUN_NAME=sb3_stun johnson163/pandia_receiver'
        os.system(cmd)
        print(cmd)
        self.receiver_container = self.docker_client.containers.get(f'sb3_receiver_{cid}') # type: ignore
        self.receiver_container_ip = self.receiver_container.attrs['NetworkSettings']['Networks']['sb3_net']['IPAddress']
        cmd = f'docker run -d --rm --network sb3_net --name sb3_sender_{cid} --hostname sb3_sender_{cid} '\
              f'--cap-add=NET_ADMIN --env NVIDIA_DRIVER_CAPABILITIES=all '\
              f'--runtime=nvidia --gpus all '\
              f'-v /tmp/pandia:/tmp '\
              f'--env OBS_PORT={self.obs_port} --env OBS_HOST=195.148.124.151 '\
              f'--env RECEIVER_IP={self.receiver_container_ip} '\
              f'--env PRINT_STEP=True -e SENDER_LOG=/tmp/sender.log --env BANDWIDTH=1000-3000 '\
              f'johnson163/pandia_sender python -um sb3_client'
        os.system(cmd)
        print(cmd)
        self.sender_container = self.docker_client.containers.get(f'sb3_sender_{cid}') # type: ignore
        self.sender_container_ip = self.sender_container.attrs['NetworkSettings']['Networks']['sb3_net']['IPAddress']
        print(f'Sender container IP: {self.sender_container_ip}, receiver container IP: {self.receiver_container_ip}')
        self.container_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.container_socket.connect((self.sender_container_ip, WEBRTC_SENDER_SB3_PORT))
        time.sleep(1)

    def stop_containers(self):
        print('Stopping containers...', flush=True)
        if self.sender_container:
            os.system(f'docker stop {self.sender_container.id} > /dev/null &')
        if self.receiver_container:
            os.system(f'docker stop {self.receiver_container.id} > /dev/null &')

    def start_webrtc(self):
        print('Starting WebRTC...', flush=True)
        buf = bytearray(1)
        buf[0] = 0  
        self.container_socket.send(buf)

    def reset(self, seed=None, options=None):
        self.context = StreamingContext(monitor_durations=self.monitor_durations)
        self.obs_thread.context = self.context
        self.action_history = ActionHistory()
        self.termination_ts = 0
        self.step_count = 0
        self.last_print_ts = 0
        self.observation = Observation(self.obs_keys, self.monitor_durations, self.hisory_size)
        self.start_ts = time.time()
        return self.observation.array(), {}

    def close(self):
        self.stop_containers()
        self.obs_thread.stop()

    def step(self, action: np.ndarray):
        self.context.reset_step_context()
        act = Action.from_array(action, self.action_keys)
        self.action_history.append(act)
        buf = bytearray(Action.shm_size() + 1)
        act.write(buf)
        buf[1:] = buf[:-1]
        buf[0] = 1
        self.container_socket.send(buf)
        if self.step_count == 0:
            self.start_webrtc()

        ts = time.time()
        while not self.context.codec_initiated:
            if time.time() - ts > self.init_timeout:
                print(f"WebRTC start timeout. Startover.", flush=True)
                self.reset()
                return self.step(action)
        ts = time.time()
        if self.step_count == 0:
            self.start_ts = ts
            print(f'WebRTC is running.', flush=True)
        time.sleep(self.step_duration)

        self.observation.append(self.context.monitor_blocks, act)
        truncated = time.time() - self.start_ts > self.duration
        r = reward(self.context)

        if self.print_step:
            if time.time() - self.last_print_ts > self.print_period:
                self.last_print_ts = time.time()
                print(f'[{self.cid}] #{self.step_count}@{int((time.time() - self.start_ts))}s '
                      f'R.w.: {r:.02f}, Act.: {act}Obs.: {self.observation}', flush=True)
        self.step_count += 1
        terminated = self.termination_ts > 0 and self.context.last_ts > self.termination_ts
        return self.observation.array(), r, terminated, truncated, {}


setup_containers()
tune.register_env('WebRTContainerEnv', lambda config: WebRTContainerEnv(**config))
gymnasium.register('WebRTContainerEnv', entry_point='pandia.agent.env_container:WebRTContainerEnv', nondeterministic=True)


def test():
    num_envs = 2
    envs = gymnasium.vector.make("WebRTContainerEnv", num_envs=num_envs)
    envs.reset()
    try:
        while True:
            action = Action(ENV_CONFIG['action_keys'])
            action.bitrate = 1024
            _, _, terminated, truncated, _ = envs.step([action.array()] * num_envs)
            if np.any(terminated) or np.any(truncated):
                break
    except KeyboardInterrupt:
        pass
    envs.close()
    # output_dir = os.path.join(RESULTS_PATH, 'env_container')
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # analyze_stream(env.context, output_dir=output_dir)
    exit(0)


if __name__ == '__main__':
    test()
