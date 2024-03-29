import json
from multiprocessing import shared_memory
import os
import socket
import subprocess
import time
from pandia.agent.action import Action
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.utils import sample
from pandia.constants import K, M, WEBRTC_RECEIVER_CONTROLLER_PORT, WEBRTC_SENDER_SB3_PORT

def parse_rangable(value):
    if type(value) is str and '-' in value:
        return [v for v in value.split('-')]
    else:
        return value


class ClientProtocol():
    def __init__(self) -> None:
        super().__init__()
        self.shm = shared_memory.SharedMemory(name="pandia", create=True, 
                                              size=Action.shm_size())
        self.process = None
        self.bw = parse_rangable(os.getenv('BANDWIDTH', 10 * M))
        self.delay = parse_rangable(os.getenv('DELAY', 0))
        self.loss = parse_rangable(os.getenv('LOSS', 0))
        self.height = os.getenv('WIDTH', 2160)
        self.fps = int(os.getenv('FPS', 30))
        self.obs_socket_path = os.getenv('OBS_SOCKET_PATH', '')
        self.ctrl_socket_path = os.getenv('CTRL_SOCKET_PATH', '')
        self.logging_path = os.getenv('LOGGING_PATH', '')
        self.sb3_logging_path = os.getenv('SB3_LOGGING_PATH', '')
        if os.path.exists(self.logging_path):
            os.remove(self.logging_path)
        if os.path.exists(self.sb3_logging_path):
            os.remove(self.sb3_logging_path)
        print(f'bw: {self.bw}, delay: {self.delay}, loss: {self.loss}, '
              f'obs_socket_path: {self.obs_socket_path}, '
              f'ctrl_socket_path: {self.ctrl_socket_path}', flush=True)


    def start_simulator(self, bw, delay, loss):
        bw = int(bw / K)
        delay = int(delay * 1000)
        loss = int(loss * 100)
        print(f'Start sender with bandwidth: {bw} kbps, delay {delay} ms', flush=True)
        os.system(f"tc qdisc del dev lo root 2> /dev/null")
        os.system(f"tc qdisc add dev lo root handle 1: netem delay {delay}ms")
        os.system(f"tc qdisc add dev lo parent 1: handle 2: tbf rate {bw}kbit burst 1500 latency 100ms")
        if self.sb3_logging_path:
            log_file = open(self.sb3_logging_path, 'w')
        else:
            log_file = subprocess.DEVNULL
        self.process = \
            subprocess.Popen(['/app/simulation_Release',
                              '--obs_socket', self.obs_socket_path,
                              '--resolution', str(self.height), '--fps', str(self.fps),
                              '--logging_path', self.logging_path,
                              '--force_fieldtrials=WebRTC-FlexFEC-03-Advertised/Enabled/WebRTC-FlexFEC-03/Enabled/WebRTC-FrameDropper/Disabled', 
                              '--path', '/app/media'],
                              stdout=log_file, stderr=log_file, shell=False)
    
    def datagram_received(self, data: bytes, addr) -> None:
        # Kill the simulation 
        if data[0] == 0:
            print(f'[{time.time()}] Received kill command', flush=True)
            if self.process:
                self.process.kill()
        # Send the action
        elif data[0] == 1:
            data = data[1:]
            data_str = ''.join('{:02x}'.format(x) for x in data[:16])
            print(f'[{time.time()}] Received action: {data_str}', flush=True)
            assert len(data) == len(self.shm.buf), f'Invalid action size: {len(data)} != {len(self.shm.buf)}'
            self.shm.buf[:] = data[:]
        elif data[0] == 2:
            config = json.loads(data[1:].decode()) if len(data) > 1 else {}
            print(f'[{time.time()}] Received start command: {config}', flush=True)
            self.start_simulator(config['bw'], config['delay'], config['loss'])
        else:
            print(f'Unknown command: {data[0]}', flush=True)


def main():
    client = ClientProtocol()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    ctrl_sock_path = client.ctrl_socket_path
    print(f'Connecting to {ctrl_sock_path}...', flush=True)
    sock.bind(ctrl_sock_path)
    os.chmod(ctrl_sock_path, 0o777)
    while True:
        data, addr = sock.recvfrom(1024)
        client.datagram_received(data, addr)



if __name__ == '__main__':
    main()
