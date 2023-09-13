import asyncio
from datetime import datetime
from multiprocessing import shared_memory
import os
import socket
import subprocess
import time
import requests
from pandia.agent.action import Action
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.utils import sample
from pandia.constants import WEBRTC_RECEIVER_CONTROLLER_PORT, WEBRTC_SENDER_SB3_PORT

TS = time.time()

def parse_rangable_int(value):
    if type(value) is str and '-' in value:
        return [int(v) for v in value.split('-')]
    else:
        return int(value)


class ClientProtocol():
    def __init__(self) -> None:
        super().__init__()
        self.shm = shared_memory.SharedMemory(name="pandia", create=True, 
                                              size=Action.shm_size())
        self.process_sender = None
        self.bw = parse_rangable_int(os.getenv('BANDWIDTH', 3000))
        self.delay = parse_rangable_int(os.getenv('DELAY', 0))
        self.loss = parse_rangable_int(os.getenv('LOSS', 0))
        self.height = os.getenv('WIDTH', 2160)
        self.fps = int(os.getenv('FPS', 30))
        self.receiver_ip = os.getenv('RECEIVER_IP', '127.0.0.1')
        self.cid = socket.gethostname().split('_')[-1]

    def reset_receiver(self):
        delay = sample(self.delay)
        print(f'Reset receiver with delay: {delay} ms', flush=True)
        r = requests.post(f'http://{self.receiver_ip}:{WEBRTC_RECEIVER_CONTROLLER_PORT}/reset', json={'latency': delay})
        print(f'Reset received: {r.text}, wait for 1s...', flush=True)
        time.sleep(1)

    def stop_sender(self):
        if self.process_sender and self.process_sender.poll() is None:
            self.process_sender.kill()

    def obs_socket_path(self):
        return f'/tmp/sockets/{self.cid}'

    def start_sender(self):
        bw = sample(self.bw)
        print(f'Start sender with bandwidth: {bw} kbps', flush=True)
        os.system(f"tc qdisc del dev eth0 root 2> /dev/null")
        os.system(f"tc qdisc add dev eth0 root tbf rate {bw}kbit burst 1000kb minburst 1540 latency 250ms")
        ts_str = datetime.now().strftime("%m_%d_%H_%M_%S")
        log_file = open(f'/tmp/{ts_str}_{socket.gethostname()}.log', 'w')
        self.process_sender = \
            subprocess.Popen(['/app/peerconnection_client_headless',
                              '--server', self.receiver_ip,
                              '--obs_socket', self.obs_socket_path(),
                              '--width', str(self.height), '--fps', str(self.fps),
                              '--force_fieldtrials=WebRTC-FlexFEC-03-Advertised/Enabled/WebRTC-FlexFEC-03/Enabled/', 
                              '--path', '/app/media'],
                              stdout=log_file, stderr=log_file, shell=False)
    
    def datagram_received(self, data: bytes, addr) -> None:
        # Reset the sender
        if data[0] == 0:
            print(f'[{time.time() - TS:.02f}] Received reset command', flush=True)
            self.stop_sender()
            self.reset_receiver()
            self.start_sender()
        # Send the action
        elif data[0] == 1:
            data = data[1:]
            data_str = ''.join('{:02x}'.format(x) for x in data[:16])
            print(f'[{time.time() - TS:.02f}] Received action: {data_str}', flush=True)
            assert len(data) == len(self.shm.buf), f'Invalid action size: {len(data)} != {len(self.shm.buf)}'
            self.shm.buf[:] = data[:]
        else:
            print(f'Unknown command: {data[0]}', flush=True)

    def ctrl_socket_path(self):
        return f'/tmp/sockets/{self.cid}_ctrl'


def main():
    client = ClientProtocol()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    ctrl_sock_path = client.ctrl_socket_path()
    print(f'[{time.time() - TS:.02f}] Connecting to {ctrl_sock_path}...', flush=True)
    sock.bind(ctrl_sock_path)
    os.chmod(ctrl_sock_path, 0o777)
    while True:
        data, addr = sock.recvfrom(1024)
        client.datagram_received(data, addr)



if __name__ == '__main__':
    main()
