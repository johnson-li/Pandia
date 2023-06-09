import argparse
from multiprocessing import shared_memory
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from pandia import BIN_PATH, RESULTS_PATH, SCRIPTS_PATH
from pandia.log_analyzer import main as main_analyzer
from pandia.log_analyzer_sender import StreamingContext, analyze_stream, parse_line, main as main_sender
from pandia.log_analyzer_receiver import Stream, parse_line as parse_line_receiver, analyze as analyze_receiver, main as main_receiver
from pandia.log_analyzer_hybrid import main as main_hybrid
from pandia.ntp.ntpclient import ntp_sync


CLIENT_ID = 18
PORT = 7000 + CLIENT_ID
SHM_NAME = f'pandia_{PORT}'
DURATION = 30
NETWORK = {
    'bw': 1024 * 1024,
    'delay': 5,
    'loss': 5,
}
SOURCE = {
    'width': 2160,
    'fps': 60,
}
ACTION = {
    # 'pacing_rate': 1000 * 1024,  # in kbps
    # 'bitrate': 10 * 1024,  # in kbps
    # 'width': 1440,
    # 'fps': 30,
    # 'fec_key': int(255 * .5),
    # 'fec_delta': int(255 * .5),
}
RESULT_DIR = os.path.join(RESULTS_PATH, "eval_static")
SENDER_LOG = 'eval_sender.log'
RECEIVER_LOG = 'eval_receiver.log'


def init_webrtc(duration=DURATION):
    shm = None
    shm_size = 10 * 4
    print(f"[{CLIENT_ID}] Initializing shm {SHM_NAME} for WebRTC")
    try:
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=shm_size)
    except FileExistsError:
        shm = \
            shared_memory.SharedMemory(name=SHM_NAME, create=False, size=shm_size)
    process = subprocess.Popen([os.path.join(SCRIPTS_PATH, 'start_webrtc_receiver_remote.sh'), 
                        '-p', str(PORT), '-d', str(duration + 5),
                        '-l', f'/tmp/{RECEIVER_LOG}'], shell=False)
    process.wait()
    # time.sleep(1)
    return shm
    

def start_webrtc():
    process_traffic_control = \
        subprocess.Popen([os.path.join(SCRIPTS_PATH, 'start_traffic_control_remote.sh'),
                            '-p', str(PORT), '-b', str(NETWORK["bw"]), '-l', str(NETWORK["loss"]),
                            '-d', str(NETWORK["delay"]),])
    process_traffic_control.wait()
    process_sender = subprocess.Popen([os.path.join(BIN_PATH, 'peerconnection_client_headless'),
                                            '--server', '195.148.127.230',
                                            '--port', str(PORT), '--name', 'sender',
                                            '--width', str(SOURCE["width"]), 
                                            '--fps', str(SOURCE["fps"]), 
                                            '--autocall', 'true',
                                            '--force_fieldtrials=WebRTC-FlexFEC-03-Advertised/Enabled/WebRTC-FlexFEC-03/Enabled/WebRTC-FrameDropper/Disabled'],
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    return process_sender


def sync_clock():
    return ntp_sync()


# Copied from env.py
def write_shm(shm, action=ACTION) -> None:
    def write_int(value, offset):
        if isinstance(value, np.ndarray):
            value = value[0]
        value = int(value)
        bytes = value.to_bytes(4, byteorder='little')
        shm.buf[offset * 4:offset * 4 + 4] = bytes
    write_int(action.get('bitrate', 0), 0)
    write_int(action.get('pacing_rate', 0), 1)
    write_int(action.get('fps', 0), 2)
    write_int(action.get('fec_key', 256), 3) # Unlike the other actions, fec values larger than 255 are regarded as invalid
    write_int(action.get('fec_delta', 256), 4)
    # write_int(padding_rate, 5)
    write_int(action.get('width', 0), 6)


def run_exp(action=ACTION, result_dir=RESULT_DIR, duration=DURATION):
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    shm = init_webrtc(duration)
    write_shm(shm, action)
    ntp_resp = sync_clock()
    process_sender = start_webrtc()
    std_out = process_sender.stdout
    start_ts = time.time()
    print(f'Running sender, wait for {duration} seconds')
    with open(os.path.join(result_dir, SENDER_LOG), 'w') as f:
        f.write(f'NTP response: precision: {ntp_resp.precision}'
                f', offset: {ntp_resp.offset}, rtt: {ntp_resp.delay}\n')
        while time.time() - start_ts < duration:
            line = std_out.readline().decode().strip()
            if line:
                f.write(f'{line}\n')
    process_sender.kill()
    print(f'Finished running sender, wait for log dump to finish')
    os.system(f'scp mobix:/tmp/{RECEIVER_LOG} {result_dir} > /dev/null')


def analyze(result_dir=RESULT_DIR):
    print(f'Analyzing logs...')
    main_analyzer(result_dir)


def main():
    parser = argparse.ArgumentParser(description="Evaluate WebRTC with static actions")
    parser.add_argument('-d', '--dry', action='store_true', help='If set, reuse existing logs')
    parser.add_argument('-r', '--repeat', type=int, default=1, help='Repeat the run of N times')
    args = parser.parse_args()
    for i in range(args.repeat):
        if not args.dry:
            run_exp(action=ACTION, result_dir=RESULT_DIR, duration=DURATION)
        analyze(RESULT_DIR)


if __name__ == "__main__":
    main()