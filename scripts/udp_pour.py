import time
import socket


def send_udp_packet(target_ip, target_port, message):
    # Create a UDP socket
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    start_ts = time.time()
    bytes = 0
    bw = 100 * 1024 * 1024

    try:
        # Send the UDP packet
        while True:
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


# Set the target IP and port
target_ip = "10.200.18.2"
target_port = 12344

# Set the message to send
message = "a" * 1400

# Send the UDP packet
send_udp_packet(target_ip, target_port, message)
