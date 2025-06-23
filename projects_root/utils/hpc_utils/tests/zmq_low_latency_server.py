import zmq
import time

class LowLatencyZMQServer:
    def __init__(self, port=10051):
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.REP)
        
        # CRITICAL: Low-latency socket options  
        self.socket.setsockopt(zmq.LINGER, 0)           # Close immediately
        self.socket.setsockopt(zmq.RCVBUF, 1024)       # Small receive buffer
        self.socket.setsockopt(zmq.SNDBUF, 1024)       # Small send buffer
        self.socket.setsockopt(zmq.RCVTIMEO, -1)       # No receive timeout
        self.socket.setsockopt(zmq.SNDTIMEO, 1000)     # 1s send timeout
        
        # Bind
        self.socket.bind(f"tcp://0.0.0.0:{port}")
        
        print(f"🚀 Low-Latency ZMQ Server listening on port {port}")
        print("Socket optimizations applied:")
        print("  - LINGER=0 (immediate close)")
        print("  - Small buffers (1KB)")
        print("  - Minimal timeouts")
        
        self.message_count = 0
        self.start_time = time.time()
        self.process_times = []
    
    def handle_request(self):
        """Handle single request with minimal overhead"""
        # Receive request
        message = self.socket.recv()
        
        # Minimal processing - just echo back
        # In your real application, put your actual logic here
        process_start = time.perf_counter()
        
        # Your computation goes here
        response = b"pong"
        
        process_end = time.perf_counter()
        process_time = (process_end - process_start) * 1000000  # Convert to microseconds
        
        # Send response immediately
        self.socket.send(response)
        
        self.message_count += 1
        self.process_times.append(process_time)
        
        # Log every 100 messages
        if self.message_count % 100 == 0:
            elapsed = time.time() - self.start_time
            rate = self.message_count / elapsed
            avg_process = sum(self.process_times[-100:]) / len(self.process_times[-100:])
            
            print(f"Processed {self.message_count} requests, "
                  f"Rate: {rate:.1f} req/sec, "
                  f"Avg process time: {avg_process:.1f}μs")
    
    def run(self):
        """Main server loop optimized for low latency"""
        print("Server running... Press Ctrl+C to stop")
        print("Optimized for minimal per-request latency")
        
        try:
            while True:
                self.handle_request()
                
        except KeyboardInterrupt:
            elapsed = time.time() - self.start_time
            rate = self.message_count / elapsed if elapsed > 0 else 0
            avg_process = sum(self.process_times) / len(self.process_times) if self.process_times else 0
            
            print(f"\n📊 Server Statistics:")
            print(f"  Total requests: {self.message_count}")
            print(f"  Runtime: {elapsed:.2f}s")
            print(f"  Average rate: {rate:.1f} req/sec")
            print(f"  Average process time: {avg_process:.1f}μs")
            print(f"  Min process time: {min(self.process_times):.1f}μs")
            print(f"  Max process time: {max(self.process_times):.1f}μs")
            
        finally:
            self.socket.close()
            self.ctx.term()

if __name__ == "__main__":
    server = LowLatencyZMQServer()
    server.run() 