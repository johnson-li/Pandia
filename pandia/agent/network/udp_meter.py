import socket
import numpy as np
import time


def receive_data():
    UDP_IP = ""  # Leave empty to listen on all available interfaces
    UDP_PORT = 12345
    BUFFER_SIZE = 10240
    duration = 1

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(0)
    sock.bind((UDP_IP, UDP_PORT))

    start_time = time.time()
    total_bytes = 0
    delays = []
    ticks = []

    while True:
        try:
            data, addr = sock.recvfrom(BUFFER_SIZE)
            total_bytes += len(data)
            tick = int.from_bytes(data[:8], 'big')
            ts = int.from_bytes(data[8:16], 'big') / 1000
            delays.append((time.time() - ts) * 1000)
            ticks.append(tick)
        except Exception:
            pass
        elapsed_time = time.time() - start_time
        if elapsed_time > duration:
            incoming_rate = total_bytes / elapsed_time  # Bytes per second
            if not delays:
                delays = [-1]
            print(f"Incoming data rate: {incoming_rate * 8 / 1024 / 1024:.2f} mbps, "
                  f"loss rate: {100 - len(ticks) / (np.max(ticks) - np.min(ticks) + 1) * 100 if ticks else -1:.02f}%, "
                  f"delay: {np.percentile(delays, 10):.02f}, {np.percentile(delays, 50):.02f}, {np.percentile(delays, 90):.02f} ms", flush=True)

            # Reset counters
            start_time = time.time()
            total_bytes = 0
            delays.clear()
            ticks.clear()


if __name__ == '__main__':
    receive_data()

