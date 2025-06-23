import zmq
import time

ctx = zmq.Context()
socket = ctx.socket(zmq.ROUTER)  # ROUTER for asynchronous load balancing
socket.bind("tcp://0.0.0.0:10051")

print("ZMQ ROUTER server listening on port 10051...")
print("Ready for DEALER clients (with proper ready signal handshake)")

message_count = 0
workers_ready = {}

while True:
    try:
        # ROUTER receives from DEALER: [dealer_id, message]
        parts = socket.recv_multipart()
        
        if len(parts) >= 2:
            dealer_id = parts[0]
            message = parts[1]
            
            # Handle "ready" signals from workers
            if message == b"ready":
                workers_ready[dealer_id] = True
                print(f"Worker {dealer_id} is ready")
                
                # Send work to ready worker
                socket.send_multipart([dealer_id, b"work_task"])
                print(f"Sent work to {dealer_id}")
                
            else:
                message_count += 1
                print(f"Received message {message_count}: {message} from {dealer_id}")
                
                # Send response back to dealer
                socket.send_multipart([dealer_id, b"pong"])
                print(f"Sent pong to {dealer_id}")
        
        else:
            print(f"Invalid message format: {parts}")
            
    except KeyboardInterrupt:
        break

socket.close()
ctx.term()