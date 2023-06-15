import socket
import time

def udp_echo_client(server_ip, server_port, message):
    # Create a UDP socket
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Send the message to the server
    start_time = time.time()
    udp_socket.sendto(message.encode(), (server_ip, server_port))
    print(f"Sent message to {server_ip}:{server_port}: {message}")

    # Receive the echoed message from the server
    data, server_address = udp_socket.recvfrom(1024)
    end_time = time.time()
    rtt = end_time - start_time

    print(f"Received echoed message from {server_address}: {data.decode()}")
    print(f"RTT: {rtt} seconds")

    # Close the socket
    udp_socket.close()

# Set the server IP and port
server_ip = "10.200.18.2"
server_port = 12345

# Set the message to send
message = "Hello, UDP echo server!"

# Send the message to the UDP echo server and measure RTT
udp_echo_client(server_ip, server_port, message)

