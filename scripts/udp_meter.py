import socket
import time


def receive_data():
    UDP_IP = ""  # Leave empty to listen on all available interfaces
    UDP_PORT = 12344
    BUFFER_SIZE = 10240
    duration = .1

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(0)
    sock.bind((UDP_IP, UDP_PORT))

    start_time = time.time()
    total_bytes = 0

    while True:
        try:
            data, addr = sock.recvfrom(BUFFER_SIZE)
            total_bytes += len(data)

        except Exception:
            pass
        elapsed_time = time.time() - start_time
        if elapsed_time > duration:
            incoming_rate = total_bytes / elapsed_time  # Bytes per second
            print(f"Incoming data rate: {incoming_rate * 8 / 1024 / 1024:.2f} mbps")

            # Reset counters
            start_time = time.time()
            total_bytes = 0


if __name__ == '__main__':
    receive_data()

