import socket
import time

# Test UDP connectivity step by step
remote_ip = "132.72.65.119"
udp_port = 10052

print(f"🔍 UDP Connectivity Debug for {remote_ip}:{udp_port}")
print("=" * 50)

# Test 1: Can we create a UDP socket?
try:
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("✅ UDP socket created successfully")
except Exception as e:
    print(f"❌ Failed to create UDP socket: {e}")
    exit(1)

# Test 2: Set a very short timeout for quick testing
client_socket.settimeout(2.0)
print("✅ Timeout set to 2 seconds")

# Test 3: Try to send a packet (no receive yet)
try:
    print(f"📤 Sending test packet to {remote_ip}:{udp_port}...")
    client_socket.sendto(b"test", (remote_ip, udp_port))
    print("✅ Packet sent successfully (no errors)")
except Exception as e:
    print(f"❌ Failed to send packet: {e}")
    client_socket.close()
    exit(1)

# Test 4: Try to receive response
try:
    print("📥 Waiting for response...")
    data, server = client_socket.recvfrom(1024)
    print(f"✅ Received response: {data} from {server}")
except socket.timeout:
    print("⏱️  No response received (timeout)")
    print("This suggests:")
    print("   - Server might not be running")
    print("   - Firewall blocking UDP traffic")
    print("   - Network routing issues")
except Exception as e:
    print(f"❌ Error receiving: {e}")

client_socket.close()

# Test 5: Alternative diagnostic - try a different port
print("\n🔍 Testing if ANY UDP traffic works...")
test_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
test_socket.settimeout(1.0)

# Try a few common ports to see if UDP works at all
test_ports = [53, 123, 10051, 10053]  # DNS, NTP, your ZMQ ports
for port in test_ports:
    try:
        test_socket.sendto(b"test", (remote_ip, port))
        print(f"✅ Could send to port {port}")
    except Exception as e:
        print(f"❌ Cannot send to port {port}: {e}")

test_socket.close()
print("\n💡 Recommendations:")
print("1. Verify UDP server is actually running on remote machine")
print("2. Check if university firewall blocks UDP")
print("3. Try running 'netstat -an | grep 10052' on server")
print("4. Consider VPN might filter UDP differently than TCP") 