import socket
import time

# UDP Server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
port = 10053
server_socket.bind(('0.0.0.0', port))  # Use different port that might work

print(f"UDP Server listening on port {port}...")
print("Waiting for UDP packets...")

packet_count = 0
while True:
    try:
        # Receive data and client address
        print(f"[{packet_count}] Waiting for packet...")
        data, client_address = server_socket.recvfrom(1024)
        packet_count += 1
        
        print(f"[{packet_count}] Received '{data}' from {client_address}")
        
        # Echo back immediately
        server_socket.sendto(b"pong", client_address)
        print(f"[{packet_count}] Sent pong back to {client_address}")
        
    except KeyboardInterrupt:
        break

server_socket.close() 