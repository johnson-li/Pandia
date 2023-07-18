import socket
import time

# Set the server IP and port
server_ip = "10.200.18.2"
server_port = 12345

# Set the message to send
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.setblocking(False)

# Send the message to the UDP echo server and measure RTT
import time

count = 0
start_ts = time.time()
period = 1
send_record = {}
while True:
    ts = time.time()
    if (ts > start_ts + count * period):
        udp_socket.sendto(str(ts).encode(), (server_ip, server_port))
        send_record[ts] = count
        count += 1
    try:
        data, server_address = udp_socket.recvfrom(1024)
        if data:
            send_ts = float(data.decode())
            c = send_record.pop(send_ts, -1)
            print(f'[#{c} {send_ts:.02f}] RTT: {(time.time() - send_ts) * 1000:.02f} ms')
    except Exception:
        pass


