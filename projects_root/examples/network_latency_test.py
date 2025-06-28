#!/usr/bin/env python3
"""
Simple network latency test to isolate pure network performance.
"""

import zmq
import time
import pickle
import statistics

def test_network_latency(server_ip: str, server_port: int, num_tests: int = 20):
    """Test pure network latency with minimal payload."""
    
    print(f"Testing network latency to {server_ip}:{server_port}")
    print(f"Running {num_tests} ping tests...")
    
    # Initialize ZeroMQ
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
    
    try:
        socket.connect(f"tcp://{server_ip}:{server_port}")
        
        latencies = []
        
        for i in range(num_tests):
            # Create minimal ping request
            request = {
                "type": "ping",
                "args": (f"test_{i}",),
            }
            
            # Measure round-trip time
            start_time = time.time()
            
            # Send request
            pickled_request = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
            socket.send(b'U' + pickled_request)
            
            # Receive response
            response_data = socket.recv()
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to ms
            latencies.append(latency)
            
            print(f"Ping {i+1}: {latency:.2f}ms")
            time.sleep(0.1)  # Small delay between tests
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
        
        print(f"\n=== NETWORK LATENCY RESULTS ===")
        print(f"Average latency: {avg_latency:.2f}ms")
        print(f"Min latency:     {min_latency:.2f}ms")
        print(f"Max latency:     {max_latency:.2f}ms")
        print(f"Std deviation:   {std_latency:.2f}ms")
        print(f"Request size:    {len(pickled_request)} bytes")
        print(f"===============================")
        
        return avg_latency
        
    except Exception as e:
        print(f"Network test failed: {e}")
        return None
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python network_latency_test.py <server_ip> <server_port>")
        sys.exit(1)
    
    server_ip = sys.argv[1]
    server_port = int(sys.argv[2])
    
    test_network_latency(server_ip, server_port) 