import socket
import time

remote_ip = "132.72.65.119"  # Your server IP
port = 10053
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Set timeout to avoid hanging
client_socket.settimeout(15.0)

print("🚀 UDP RTT Benchmark")

# Warm-up
try:
    client_socket.sendto(b"ping", (remote_ip, port))
    data, server = client_socket.recvfrom(1024)
    print("UDP connection established")
except Exception as e:
    print(f"UDP connection failed: {e}")
    exit(1)

latencies = []
failed_packets = 0

for i in range(1000):
    try:
        start = time.perf_counter()
        client_socket.sendto(b"ping", (remote_ip, port))
        data, server = client_socket.recvfrom(1024)
        end = time.perf_counter()
        
        rtt = (end - start) * 1000
        latencies.append(rtt)
        
        if i % 100 == 0:
            print(f"Completed {i}/1000 packets...")
            
    except socket.timeout:
        failed_packets += 1
        print(f"Packet {i} timed out")
    except Exception as e:
        failed_packets += 1
        print(f"Packet {i} failed: {e}")

client_socket.close()

if latencies:
    avg = sum(latencies) / len(latencies)
    print(f"\n📊 UDP Benchmark Results:")
    print(f"Avg RTT: {avg:.3f} ms")
    print(f"Min RTT: {min(latencies):.3f} ms") 
    print(f"Max RTT: {max(latencies):.3f} ms")
    print(f"Packet loss: {failed_packets}/1000 ({100*failed_packets/1000:.1f}%)")
    print(f"Successful packets: {len(latencies)}")
else:
    print("❌ All packets failed!") 