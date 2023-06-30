from multiprocessing import shared_memory
import os
import subprocess
import time

import numpy as np
from pandia import BIN_PATH, RESULTS_PATH, SCRIPTS_PATH
from pandia.log_analyzer import StreamingContext, analyze_stream, parse_line
from pandia.log_analyzer_receiver import Stream, parse_line as parse_line_receiver, analyze as analyze_receiver


GCC = False
CLIENT_ID = 18
PORT = 7000 + CLIENT_ID
SHM_NAME = f'pandia_{PORT}'
DURATION = 10
BW = 1024 * 1024
PACING_RATE = 1024 * 1024
BITRATE = 2 * 1024
WIDTH = 1080
DELAY = 0
FPS = 30
RESULT_DIR = os.path.join(RESULTS_PATH, "eval_static")
SENDER_LOG = 'eval_sender.log'
RECEIVER_LOG = 'eval_receiver.log'


def init_webrtc():
    shm = None
    shm_size = 10 * 4
    if not GCC:
        print(f"[{CLIENT_ID}] Initializing shm {SHM_NAME} for WebRTC")
        try:
            shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=shm_size)
        except FileExistsError:
            shm = \
                shared_memory.SharedMemory(name=SHM_NAME, create=False, size=shm_size)
    process = subprocess.Popen([os.path.join(SCRIPTS_PATH, 'start_webrtc_receiver_remote.sh'), 
                        '-p', str(PORT), '-d', str(DURATION + 3), 
                        '-l', f'/tmp/{RECEIVER_LOG}'], shell=False)
    process.wait()
    time.sleep(1)
    return shm
    

def start_webrtc():
    process_traffic_control = \
        subprocess.Popen([os.path.join(SCRIPTS_PATH, 'start_traffic_control_remote.sh'),
                            '-p', str(PORT), '-b', str(BW), '-d', str(DELAY),])
    process_traffic_control.wait()
    process_sender = subprocess.Popen([os.path.join(BIN_PATH, 'peerconnection_client_headless'),
                                            '--server', '195.148.127.230',
                                            '--port', str(PORT), '--name', 'sender',
                                            '--width', str(WIDTH), '--fps', str(FPS), '--autocall', 'true',
                                            '--force_fieldtrials=WebRTC-FlexFEC-03-Advertised/Enabled/WebRTC-FlexFEC-03/Enabled/'],
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    return process_sender


# Copied from env.py
def write_shm(shm) -> None:
    def write_int(value, offset):
        if isinstance(value, np.ndarray):
            value = value[0]
        value = int(value)
        bytes = value.to_bytes(4, byteorder='little')
        shm.buf[offset * 4:offset * 4 + 4] = bytes
    write_int(BITRATE, 0)
    write_int(PACING_RATE, 1)
    write_int(FPS, 2)
    # write_int(fec_rate_key, 3)
    # write_int(fec_rate_delta, 4)
    # write_int(padding_rate, 5)
    write_int(WIDTH, 6)


def run():
    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)
    shm = init_webrtc()
    if shm:
        write_shm(shm)
    process_sender = start_webrtc()
    std_out = process_sender.stdout
    start_ts = time.time()
    print(f'Running sender, wait for {DURATION} seconds')
    with open(os.path.join(RESULT_DIR, SENDER_LOG), 'w') as f:
        while time.time() - start_ts < DURATION:
            line = std_out.readline().decode().strip()
            if line:
                f.write(f'{line}\n')
    os.system(f'scp mobix:/tmp/{RECEIVER_LOG} {RESULT_DIR} > /dev/null')


def analyze():
    context = StreamingContext()
    for line in open(os.path.join(RESULT_DIR, SENDER_LOG)).readlines():
        parse_line(line, context)
    analyze_stream(context, RESULT_DIR)
    stream = Stream()
    for line in open(os.path.join(RESULT_DIR, RECEIVER_LOG)).readlines():
        parse_line_receiver(line, stream)
    analyze_receiver(stream, RESULT_DIR)


def main():
    run()
    analyze()


if __name__ == "__main__":
    main()