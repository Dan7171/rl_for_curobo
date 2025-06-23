import zmq
import time

class LowLatencyZMQClient:
    def __init__(self, server_ip, port=10051):
        self.server_ip = server_ip
        self.port = port
        self.ctx = zmq.Context()
        
        # Create optimized socket
        self.socket = self.ctx.socket(zmq.REQ)
        
        # CRITICAL: Low-latency socket options
        self.socket.setsockopt(zmq.LINGER, 0)           # Close immediately
        self.socket.setsockopt(zmq.IMMEDIATE, 1)        # Don't queue on disconnected peers
        self.socket.setsockopt(zmq.RCVBUF, 1024)       # Small receive buffer
        self.socket.setsockopt(zmq.SNDBUF, 1024)       # Small send buffer
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)     # 5s timeout
        self.socket.setsockopt(zmq.SNDTIMEO, 5000)     # 5s send timeout
        
        # Connect
        self.socket.connect(f"tcp://{self.server_ip}:{self.port}")
        print(f"🚀 Low-latency ZMQ client connected to {self.server_ip}:{self.port}")
    
    def send_request(self, message=b"ping"):
        """Send a single optimized request"""
        start = time.perf_counter()
        
        # Send minimal payload
        self.socket.send(message)
        
        # Receive response
        response = self.socket.recv()
        
        end = time.perf_counter()
        return (end - start) * 1000  # Convert to ms
    
    def benchmark_latency(self, num_requests=1000):
        """Benchmark latency with optimized settings"""
        print(f"🔥 Low-Latency Benchmark ({num_requests} requests)")
        print("Socket optimizations applied:")
        print("  - LINGER=0 (immediate close)")
        print("  - IMMEDIATE=1 (no queueing)")
        print("  - Small buffers (1KB)")
        print("  - Timeouts set")
        
        latencies = []
        failed = 0
        
        # Warm-up
        try:
            for _ in range(5):
                self.send_request()
            print("✅ Warm-up completed")
        except Exception as e:
            print(f"❌ Warm-up failed: {e}")
            return
        
        # Main benchmark
        print("Starting latency test...")
        
        for i in range(num_requests):
            try:
                latency = self.send_request()
                latencies.append(latency)
                
                if (i + 1) % 100 == 0:
                    current_avg = sum(latencies[-100:]) / len(latencies[-100:])
                    print(f"  {i+1}/{num_requests} - Last 100 avg: {current_avg:.2f}ms")
                    
            except Exception as e:
                failed += 1
                if failed < 5:  # Only print first few errors
                    print(f"  Request {i} failed: {e}")
        
        # Results
        if latencies:
            avg_lat = sum(latencies) / len(latencies)
            min_lat = min(latencies)
            max_lat = max(latencies)
            
            # Calculate percentiles
            sorted_lat = sorted(latencies)
            p50 = sorted_lat[len(sorted_lat)//2]
            p95 = sorted_lat[int(len(sorted_lat)*0.95)]
            p99 = sorted_lat[int(len(sorted_lat)*0.99)]
            
            print(f"\n📊 Low-Latency Results:")
            print(f"  Average: {avg_lat:.2f}ms")
            print(f"  Minimum: {min_lat:.2f}ms")
            print(f"  Maximum: {max_lat:.2f}ms")
            print(f"  50th percentile: {p50:.2f}ms")
            print(f"  95th percentile: {p95:.2f}ms") 
            print(f"  99th percentile: {p99:.2f}ms")
            print(f"  Success rate: {len(latencies)}/{num_requests}")
            print(f"  Failed requests: {failed}")
            
            # Compare to baseline
            baseline = 11.26
            improvement = ((baseline - avg_lat) / baseline) * 100
            if avg_lat < baseline:
                print(f"  🎉 Improvement: {improvement:.1f}% faster than baseline!")
            else:
                print(f"  📈 Change: {improvement:.1f}% vs baseline")
        
        return latencies
    
    def close(self):
        """Clean shutdown"""
        self.socket.close()
        self.ctx.term()

if __name__ == "__main__":
    remote_ip = "132.72.65.119"
    
    client = LowLatencyZMQClient(remote_ip)
    try:
        client.benchmark_latency(1000)
    finally:
        client.close() 