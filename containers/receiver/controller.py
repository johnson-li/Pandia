import http.server
import json
import os
import socket
import socketserver
from subprocess import DEVNULL, PIPE, Popen


PROCESS = None

def log(msg):
    print(msg, flush=True)

def start_receiver(data={}):
    global PROCESS
    if PROCESS:
        PROCESS.kill()
    log(f'Reset receiver: {data}')
    stun_ip = socket.gethostbyname('stun')
    os.system("tc qdisc del dev eth0 root 2> /dev/null")
    os.system(f"tc qdisc add dev eth0 root netem delay {data.get('delay', 0)}ms")
    PROCESS = Popen(['/app/peerconnection_client_headless', '--receiving_only'], 
                    env={'WEBRTC_CONNECT': f'stun:{stun_ip}:3478'}, 
                    stdout=DEVNULL, stderr=DEVNULL)


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
