import argparse
import time
import socket


def send_udp_packet(target_ip, target_port, message):
    # Create a UDP socket
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.setblocking(False)
    start_ts = time.time()
    sent = 0
    bw = 80 * 1024 * 1024
    start_ts = time.time()
    tick = 0
    duration = 300000000

    while time.time() - start_ts < duration:
        try:
            next_ts = sent * 8 / bw + start_ts
            if next_ts <= time.time():
                data = tick.to_bytes(8, 'big')
                data += int(time.time() * 1000).to_bytes(8, 'big')
                data += message.encode()
                sent += len(message)
                tick += 1
                udp_socket.sendto(data, (target_ip, target_port))
        except socket.error as e:
            pass


def main():
    # Set the target IP and port
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_ip", default="127.0.0.1", help="The target IP address")
    parser.add_argument("--target_port", default=12345, help="The target port")
    args = parser.parse_args()
    target_ip = args.target_ip
    target_port = args.target_port

    # Set the message to send
    message = "a" * 1400

    # Send the UDP packet
    send_udp_packet(target_ip, target_port, message)


if __name__ == "__main__":
    main()
