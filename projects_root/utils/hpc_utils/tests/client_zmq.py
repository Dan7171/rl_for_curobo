import zmq
import time

remote_ip = "132.72.65.119"  # <-- Replace with your server's IP
ctx = zmq.Context()
socket = ctx.socket(zmq.REQ)
socket.connect(f"tcp://{remote_ip}:10051")

# Warm-up
# print("Warming up...")
socket.send(b"ping")
print("Sent ping")
socket.recv()
print("Warmed up")

latencies = []
for _ in range(1000):
    start = time.perf_counter()
    # socket.send(b"ping")
    # socket.recv()
    t0 = time.time()
    socket.send(b"ping")
    print(f"Sent at {t0:.6f}")
    msg = socket.recv()
    t1 = time.time()
    print(f"Received at {t1:.6f}")
    print(f"Total RTT: {(t1 - t0)*1000:.3f} ms")
    end = time.perf_counter()
    latencies.append((end - start) * 1000)

avg = sum(latencies) / len(latencies)
print(f"\n🌐 Remote ZMQ TCP Benchmark")
print(f"Avg RTT: {avg:.3f} ms")
print(f"Min RTT: {min(latencies):.3f} ms")
print(f"Max RTT: {max(latencies):.3f} ms")