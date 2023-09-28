from datetime import datetime
import http.server
import json
import os
from random import random
import socket
import socketserver
from subprocess import DEVNULL, PIPE, Popen
import sys
import time


PROCESS = None
TS = time.time()


def log(msg):
    print(f'[{time.time() - TS:.02f}] {msg}', flush=True)


def start_receiver(data={}):
    global PROCESS
    if PROCESS and PROCESS.poll() is None:
        # If all webrtc receivers are stated at the same time, 
        # some of they may fail to listen on the serving port
        time.sleep(random())
        PROCESS.kill()
        ts = time.time()
        while PROCESS.poll() is None:
            time.sleep(0.1)
            if time.time() - ts > 3:
                log('Timeout waiting for receiver to stop')
                break
        log('Receiver stopped')
    os.system('fuser -k 9999/tcp') 

    log(f'Reset receiver: {data}')
    stun_name = os.getenv('STUN_NAME', 'stun')
    stun_ip = socket.gethostbyname(stun_name)
    obs_host = os.getenv('OBS_HOST', '')
    obs_port = os.getenv('OBS_PORT', '')
    os.system("tc qdisc del dev eth0 root 2> /dev/null")
    os.system(f"tc qdisc add dev eth0 root netem delay {data.get('delay', 0)}ms")
    ts_str = datetime.now().strftime("%m_%d_%H_%M_%S")
    log_path = f'/tmp/{ts_str}_{socket.gethostname()}.log'
    PROCESS = Popen(['/app/peerconnection_client_headless', '--receiving_only',
                     '--logging_path', log_path], 
                    env={'WEBRTC_CONNECT': f'stun:{stun_ip}:3478'},
                    stdout=DEVNULL, stderr=DEVNULL, shell=False)

    success = False
    for i in range(3):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(('localhost', 9999))
            s.close()
            success = True
            break
        except Exception as e:
            log(e)
            time.sleep(1)
    if not success:
        log('Receiver startup failed, retry')
        start_receiver(data)

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/reset':
            start_receiver()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'OK')
        else:
            log(f'Unrecognised path: {self.path}')

    def do_POST(self):
        if self.path == '/reset':
            data_str = self.rfile.read(int(self.headers['Content-Length']))
            start_receiver(json.loads(data_str))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'OK')
        else:
            log(f'Unrecognised path: {self.path}')


def main():
    start_receiver()
    PORT = 9998
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        log(f"serving at port {PORT}")
        httpd.serve_forever()

if __name__ == '__main__':
    main()
