import zmq
import time

remote_ip =  "132.72.65.119" # "132.72.180.175"
ctx = zmq.Context()
socket = ctx.socket(zmq.DEALER)  # DEALER for async communication
socket.connect(f"tcp://{remote_ip}:10051")
print(f"Connected to {remote_ip}:10051")

print("🔄 ZMQ DEALER/ROUTER Ready Signal Pattern")

# CRITICAL: Send "ready" signal first - this is the standard ZeroMQ pattern
# for DEALER/ROUTER to avoid the connection timing issue
print("Sending ready signal...")
socket.send(b"ready")

# Wait for work assignment from ROUTER
print("Waiting for work assignment...")
work = socket.recv()
print(f"Received work: {work}")

# Now do the actual ping test
print("\n=== Testing ping after ready handshake ===")
for i in range(10):
    print(f"Sending ping {i+1}")
    socket.send(b"ping")
    
    response = socket.recv()
    print(f"Received: {response}")
    
    time.sleep(0.1)

print("Test completed successfully!")
socket.close()
ctx.term() 