import socket

def udp_echo_server(server_ip, server_port):
    # Create a UDP socket
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind the socket to a specific IP address and port
    udp_socket.bind((server_ip, server_port))
    print(f"UDP echo server is listening on {server_ip}:{server_port}")

    while True:
        # Receive data from the client
        data, client_address = udp_socket.recvfrom(1024)
        print(f"Received data from {client_address}: {data.decode()}")

        # Send the data back to the client
        udp_socket.sendto(data, client_address)

    # Close the socket (unreachable in this example)
    udp_socket.close()

# Set the server IP and port
server_ip = "0.0.0.0"  # Listen on all available network interfaces
server_port = 12345

# Start the UDP echo server
udp_echo_server(server_ip, server_port)
