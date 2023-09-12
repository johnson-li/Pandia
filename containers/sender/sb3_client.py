import asyncio
from datetime import datetime
from multiprocessing import shared_memory
import os
import socket
import subprocess
import time
import numpy as np

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


class ClientProtocol(asyncio.Protocol):
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
        self.obs_host = os.getenv('OBS_HOST', '127.0.0.1')
        self.obs_port = os.getenv('OBS_PORT', 9990)

    def connection_made(self, transport) -> None:
        self.transport = transport

    def reset_receiver(self):
        delay = sample(self.delay)
        print(f'Reset receiver with delay: {delay} ms', flush=True)
        r = requests.post(f'http://{self.receiver_ip}:{WEBRTC_RECEIVER_CONTROLLER_PORT}/reset', json={'latency': delay})
        print(f'Reset received: {r.text}, wait for 1s...', flush=True)
        time.sleep(1)

    def stop_sender(self):
        if self.process_sender and self.process_sender.poll() is None:
            self.process_sender.kill()

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
                              '--obs_port', str(self.obs_port), '--obs_host', self.obs_host,
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


async def main():
    loop = asyncio.get_running_loop()
    on_con_lost = loop.create_future()
    print(f'[{time.time() - TS:.02f}] Listening on {WEBRTC_SENDER_SB3_PORT}...', flush=True)
    transport, protocol = \
        await loop.create_datagram_endpoint(lambda: ClientProtocol(), 
                                            local_addr=("0.0.0.0", WEBRTC_SENDER_SB3_PORT))
    try:
        await on_con_lost
    finally:
        transport.close()


if __name__ == '__main__':
    asyncio.run(main())
