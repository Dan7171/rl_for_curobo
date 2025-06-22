#!/usr/bin/env python3
"""
Network Diagnostics Tool for MPC Client-Server Setup

This tool helps identify where the bottleneck is:
1. Client-side network (ethernet/wifi)
2. Server-side network 
3. Client-side Python processing
4. Network latency between client and server

Usage:
    python network_diagnostics.py --server_ip <SERVER_IP> --server_port <PORT>
"""

import time
import socket
import subprocess
import sys
import argparse
import statistics
from typing import List, Tuple
import pickle
import zlib

try:
    import zmq
except ImportError:
    print("ZMQ not available - some tests will be skipped")
    zmq = None


class NetworkDiagnostics:
    """Comprehensive network diagnostics for MPC client-server setup."""
    
    def __init__(self, server_ip: str, server_port: int):
        self.server_ip = server_ip
        self.server_port = server_port
        self.results = {}
    
    def run_all_tests(self):
        """Run all diagnostic tests."""
        print("🔍 Network Diagnostics for MPC Client-Server Setup")
        print("=" * 60)
        print(f"Target: {self.server_ip}:{self.server_port}")
        print()
        
        # Test 1: Basic connectivity
        print("1️⃣  Testing basic connectivity...")
        self.test_basic_connectivity()
        
        # Test 2: Network latency (ping)
        print("\n2️⃣  Testing network latency...")
        self.test_network_latency()
        
        # Test 3: TCP connection speed
        print("\n3️⃣  Testing TCP connection speed...")
        self.test_tcp_connection_speed()
        
        # Test 4: Data transfer speed
        print("\n4️⃣  Testing data transfer speed...")
        self.test_data_transfer_speed()
        
        # Test 5: Python serialization speed
        print("\n5️⃣  Testing Python serialization speed...")
        self.test_serialization_speed()
        
        # Test 6: ZMQ performance (if available)
        if zmq:
            print("\n6️⃣  Testing ZMQ performance...")
            self.test_zmq_performance()
        
        # Summary
        print("\n" + "=" * 60)
        self.print_summary()
    
    def test_basic_connectivity(self):
        """Test if we can reach the server."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((self.server_ip, self.server_port))
            sock.close()
            
            if result == 0:
                print("✅ Server is reachable")
                self.results['connectivity'] = 'SUCCESS'
            else:
                print("❌ Cannot connect to server")
                self.results['connectivity'] = 'FAILED'
                return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            self.results['connectivity'] = 'ERROR'
            return False
        
        return True
    
    def test_network_latency(self):
        """Test network latency using ping."""
        try:
            # Use ping command to test network latency
            if sys.platform.startswith('win'):
                cmd = ['ping', '-n', '10', self.server_ip]
            else:
                cmd = ['ping', '-c', '10', self.server_ip]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Parse ping results
                lines = result.stdout.split('\n')
                times = []
                for line in lines:
                    if 'time=' in line:
                        # Extract time value
                        time_part = line.split('time=')[1].split()[0]
                        if 'ms' in time_part:
                            time_val = float(time_part.replace('ms', ''))
                            times.append(time_val)
                
                if times:
                    avg_latency = statistics.mean(times)
                    min_latency = min(times)
                    max_latency = max(times)
                    
                    print(f"📊 Ping latency: {avg_latency:.1f}ms avg, {min_latency:.1f}ms min, {max_latency:.1f}ms max")
                    self.results['ping_latency_ms'] = avg_latency
                    
                    # Analyze latency
                    if avg_latency < 1:
                        print("✅ Excellent latency (same machine/LAN)")
                    elif avg_latency < 5:
                        print("✅ Good latency (local network)")
                    elif avg_latency < 20:
                        print("⚠️  Moderate latency (campus network)")
                    else:
                        print("❌ High latency (may impact performance)")
                else:
                    print("❌ Could not parse ping results")
            else:
                print(f"❌ Ping failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("❌ Ping timeout")
        except Exception as e:
            print(f"❌ Ping error: {e}")
    
    def test_tcp_connection_speed(self):
        """Test TCP connection establishment speed."""
        connection_times = []
        
        for i in range(10):
            try:
                start_time = time.time()
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                sock.connect((self.server_ip, self.server_port))
                connect_time = (time.time() - start_time) * 1000  # Convert to ms
                sock.close()
                connection_times.append(connect_time)
            except Exception as e:
                print(f"❌ Connection {i+1} failed: {e}")
                return
        
        if connection_times:
            avg_connect = statistics.mean(connection_times)
            min_connect = min(connection_times)
            max_connect = max(connection_times)
            
            print(f"📊 TCP connection: {avg_connect:.1f}ms avg, {min_connect:.1f}ms min, {max_connect:.1f}ms max")
            self.results['tcp_connect_ms'] = avg_connect
            
            if avg_connect < 1:
                print("✅ Very fast TCP connections")
            elif avg_connect < 5:
                print("✅ Fast TCP connections")
            elif avg_connect < 20:
                print("⚠️  Moderate TCP connection speed")
            else:
                print("❌ Slow TCP connections")
    
    def test_data_transfer_speed(self):
        """Test raw data transfer speed."""
        try:
            # Create test data (similar to MPC payload)
            test_data = b'x' * 10000  # 10KB test payload
            
            transfer_times = []
            for i in range(5):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)
                sock.connect((self.server_ip, self.server_port))
                
                start_time = time.time()
                sock.send(test_data)
                response = sock.recv(1024)  # Expect some response
                transfer_time = (time.time() - start_time) * 1000
                
                sock.close()
                transfer_times.append(transfer_time)
            
            if transfer_times:
                avg_transfer = statistics.mean(transfer_times)
                print(f"📊 Data transfer (10KB): {avg_transfer:.1f}ms avg")
                self.results['data_transfer_ms'] = avg_transfer
                
                if avg_transfer < 5:
                    print("✅ Fast data transfer")
                elif avg_transfer < 20:
                    print("⚠️  Moderate data transfer speed")
                else:
                    print("❌ Slow data transfer")
                    
        except Exception as e:
            print(f"❌ Data transfer test failed: {e}")
    
    def test_serialization_speed(self):
        """Test Python serialization/deserialization speed."""
        # Create test data similar to MPC requests
        test_data = {
            'type': 'call_method',
            'args': ('mpc.step', [1.0] * 7, {'max_attempts': 1})
        }
        
        # Test pickle serialization
        pickle_times = []
        for _ in range(100):
            start_time = time.time()
            pickled = pickle.dumps(test_data, protocol=pickle.HIGHEST_PROTOCOL)
            pickle_time = (time.time() - start_time) * 1000
            pickle_times.append(pickle_time)
        
        avg_pickle = statistics.mean(pickle_times)
        print(f"📊 Pickle serialization: {avg_pickle:.3f}ms avg")
        self.results['pickle_serialize_ms'] = avg_pickle
        
        # Test pickle deserialization
        pickled_data = pickle.dumps(test_data, protocol=pickle.HIGHEST_PROTOCOL)
        unpickle_times = []
        for _ in range(100):
            start_time = time.time()
            unpickled = pickle.loads(pickled_data)
            unpickle_time = (time.time() - start_time) * 1000
            unpickle_times.append(unpickle_time)
        
        avg_unpickle = statistics.mean(unpickle_times)
        print(f"📊 Pickle deserialization: {avg_unpickle:.3f}ms avg")
        self.results['pickle_deserialize_ms'] = avg_unpickle
        
        # Test compression
        compress_times = []
        for _ in range(100):
            start_time = time.time()
            compressed = zlib.compress(pickled_data, level=1)
            compress_time = (time.time() - start_time) * 1000
            compress_times.append(compress_time)
        
        avg_compress = statistics.mean(compress_times)
        compression_ratio = len(compressed) / len(pickled_data)
        print(f"📊 Compression: {avg_compress:.3f}ms avg, ratio: {compression_ratio:.2f}")
        self.results['compression_ms'] = avg_compress
        self.results['compression_ratio'] = compression_ratio
    
    def test_zmq_performance(self):
        """Test ZMQ performance if available."""
        if zmq is None:
            print("❌ ZMQ not available")
            return
            
        try:
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
            socket.connect(f"tcp://{self.server_ip}:{self.server_port}")
            
            # Test simple request-response
            test_request = b"ping"
            zmq_times = []
            
            for i in range(5):
                try:
                    start_time = time.time()
                    socket.send(test_request)
                    response = socket.recv()
                    zmq_time = (time.time() - start_time) * 1000
                    zmq_times.append(zmq_time)
                except zmq.Again:
                    print(f"❌ ZMQ request {i+1} timed out")
                    break
            
            socket.close()
            context.term()
            
            if zmq_times:
                avg_zmq = statistics.mean(zmq_times)
                print(f"📊 ZMQ round-trip: {avg_zmq:.1f}ms avg")
                self.results['zmq_roundtrip_ms'] = avg_zmq
            
        except Exception as e:
            print(f"❌ ZMQ test failed: {e}")
    
    def print_summary(self):
        """Print diagnostic summary and recommendations."""
        print("📋 DIAGNOSTIC SUMMARY")
        print("-" * 40)
        
        # Analyze results
        issues = []
        
        # Check connectivity
        if self.results.get('connectivity') != 'SUCCESS':
            issues.append("❌ Server connectivity issues")
        
        # Check latency
        ping_latency = self.results.get('ping_latency_ms', 0)
        if ping_latency > 20:
            issues.append(f"❌ High network latency ({ping_latency:.1f}ms)")
        elif ping_latency > 5:
            issues.append(f"⚠️  Moderate network latency ({ping_latency:.1f}ms)")
        
        # Check TCP performance
        tcp_connect = self.results.get('tcp_connect_ms', 0)
        if tcp_connect > 20:
            issues.append(f"❌ Slow TCP connections ({tcp_connect:.1f}ms)")
        
        # Check serialization
        pickle_time = self.results.get('pickle_serialize_ms', 0) + self.results.get('pickle_deserialize_ms', 0)
        if pickle_time > 1:
            issues.append(f"⚠️  Slow serialization ({pickle_time:.1f}ms total)")
        
        # Print issues
        if issues:
            print("🚨 IDENTIFIED ISSUES:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("✅ No major issues detected")
        
        print("\n🎯 RECOMMENDATIONS:")
        
        # Network recommendations
        if ping_latency > 20:
            print("  • Consider using a wired connection instead of WiFi")
            print("  • Check if there's a closer server or different network route")
            print("  • Network latency is the main bottleneck (~{:.1f}ms per request)".format(ping_latency))
        
        # Performance recommendations
        if tcp_connect > 10:
            print("  • TCP connection overhead is significant")
            print("  • Consider connection pooling or persistent connections")
        
        if pickle_time > 0.5:
            print("  • Python serialization is slow")
            print("  • Consider using faster serialization (msgpack, protobuf)")
        
        # Overall assessment
        total_overhead = ping_latency + tcp_connect + pickle_time
        print(f"\n📊 ESTIMATED OVERHEAD PER MPC STEP: ~{total_overhead:.1f}ms")
        
        if ping_latency > total_overhead * 0.7:
            print("🔍 BOTTLENECK: Network latency is the main issue")
        elif tcp_connect > total_overhead * 0.3:
            print("🔍 BOTTLENECK: TCP connection overhead")
        elif pickle_time > total_overhead * 0.3:
            print("🔍 BOTTLENECK: Python serialization")
        else:
            print("🔍 BOTTLENECK: Mixed factors")


def main():
    """Main diagnostic entry point."""
    parser = argparse.ArgumentParser(description="Network Diagnostics for MPC Client-Server")
    parser.add_argument("--server_ip", type=str, required=True, help="Server IP address")
    parser.add_argument("--server_port", type=int, default=10051, help="Server port")
    args = parser.parse_args()
    
    diagnostics = NetworkDiagnostics(args.server_ip, args.server_port)
    diagnostics.run_all_tests()


if __name__ == "__main__":
    main() 