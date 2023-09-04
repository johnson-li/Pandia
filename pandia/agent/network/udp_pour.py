import argparse
import time
import socket


def send_udp_packet(target_ip, target_port, message):
    # Create a UDP socket
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    start_ts = time.time()
    bytes = 0
    bw = 100 * 1024 * 1024
    start_ts = time.time()

    try:
        # Send the UDP packet
        while time.time() - start_ts < 10:
            udp_socket.sendto(message.encode(), (target_ip, target_port))
            bytes += len(message)
            next_ts = bytes * 8 / bw + start_ts
            wait_ts = next_ts - time.time()
            wait_ts = 0
            if wait_ts > 0:
                time.sleep(wait_ts)
    except socket.error as e:
        print(f"Error: {e}")
    finally:
        # Close the socket
        udp_socket.close()


def main():
    # Set the target IP and port
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_ip", default="10.200.18.2", help="The target IP address")
    parser.add_argument("--target_port", default=12344, help="The target port")
    args = parser.parse_args()
    target_ip = args.target_ip
    target_port = args.target_port

    # Set the message to send
    message = "a" * 1400

    # Send the UDP packet
    send_udp_packet(target_ip, target_port, message)


if __name__ == "__main__":
    main()
