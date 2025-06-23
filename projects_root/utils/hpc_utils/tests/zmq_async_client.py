import zmq
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

class AsyncZMQClient:
    def __init__(self, server_ip, port=10051):
        self.server_ip = server_ip
        self.port = port
        self.ctx = zmq.Context()
        
    def create_socket(self):
        """Create a new REQ socket for each thread"""
        socket = self.ctx.socket(zmq.REQ)
        socket.connect(f"tcp://{self.server_ip}:{self.port}")
        return socket
    
    def send_single_request(self, request_id):
        """Send a single request and return the result with timing"""
        socket = self.create_socket()
        
        try:
            start_time = time.perf_counter()
            
            # Send request
            socket.send(f"ping_{request_id}".encode())
            
            # Receive response
            response = socket.recv()
            
            end_time = time.perf_counter()
            rtt = (end_time - start_time) * 1000  # Convert to ms
            
            return {
                'request_id': request_id,
                'response': response,
                'rtt_ms': rtt,
                'success': True
            }
            
        except Exception as e:
            return {
                'request_id': request_id,
                'error': str(e),
                'success': False
            }
        finally:
            socket.close()
    
    def test_sync_baseline(self, num_requests=100):
        """Traditional synchronous approach for comparison"""
        print(f"🐌 Synchronous Baseline Test ({num_requests} requests)")
        
        socket = self.create_socket()
        latencies = []
        
        start_time = time.perf_counter()
        
        for i in range(num_requests):
            try:
                req_start = time.perf_counter()
                socket.send(f"ping_{i}".encode())
                response = socket.recv()
                req_end = time.perf_counter()
                
                rtt = (req_end - req_start) * 1000
                latencies.append(rtt)
                
                if (i + 1) % 20 == 0:
                    print(f"  Completed {i + 1}/{num_requests}")
                    
            except Exception as e:
                print(f"  Request {i} failed: {e}")
        
        total_time = time.perf_counter() - start_time
        socket.close()
        
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            throughput = len(latencies) / total_time
            
            print(f"  ✅ Results:")
            print(f"     Total time: {total_time:.2f}s")
            print(f"     Avg latency: {avg_latency:.2f}ms")
            print(f"     Throughput: {throughput:.1f} req/sec")
            print(f"     Success rate: {len(latencies)}/{num_requests}")
        
        return latencies
    
    def test_async_pipelined(self, num_requests=100, max_concurrent=10):
        """Async pipelined approach using ThreadPoolExecutor"""
        print(f"🚀 Async Pipelined Test ({num_requests} requests, {max_concurrent} concurrent)")
        
        results = []
        start_time = time.perf_counter()
        
        # Use ThreadPoolExecutor to manage concurrent requests
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit all requests
            future_to_id = {
                executor.submit(self.send_single_request, i): i 
                for i in range(num_requests)
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_id):
                result = future.result()
                results.append(result)
                completed += 1
                
                if completed % 20 == 0:
                    print(f"  Completed {completed}/{num_requests}")
        
        total_time = time.perf_counter() - start_time
        
        # Analyze results
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        if successful:
            latencies = [r['rtt_ms'] for r in successful]
            avg_latency = sum(latencies) / len(latencies)
            throughput = len(successful) / total_time
            
            print(f"  ✅ Results:")
            print(f"     Total time: {total_time:.2f}s")
            print(f"     Avg latency: {avg_latency:.2f}ms")
            print(f"     Throughput: {throughput:.1f} req/sec")
            print(f"     Success rate: {len(successful)}/{num_requests}")
            
            if failed:
                print(f"     Failed requests: {len(failed)}")
        
        return successful, failed
    
    def run_comparison(self):
        """Run both sync and async tests for comparison"""
        print("🔄 ZMQ Async/Pipelined Performance Comparison")
        print("=" * 60)
        
        # Test 1: Sync baseline
        sync_results = self.test_sync_baseline(100)
        
        print()
        
        # Test 2: Async pipelined with different concurrency levels
        concurrency_levels = [5, 10, 20]
        
        for concurrency in concurrency_levels:
            print()
            async_results, failed = self.test_async_pipelined(100, concurrency)
            
        print("\n🎯 Summary:")
        print("The async/pipelined approach should show:")
        print("- Similar individual latency (~11ms)")
        print("- Much higher throughput (requests/second)")
        print("- Better resource utilization")

if __name__ == "__main__":
    remote_ip = "132.72.65.119"
    client = AsyncZMQClient(remote_ip)
    client.run_comparison() 