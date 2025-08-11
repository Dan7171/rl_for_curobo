#!/usr/bin/env python3
"""
Isaac Sim WebRTC Client
Connects to Isaac Sim's native WebRTC streaming service
"""

import asyncio
import websockets
import json
import argparse
import signal
import sys
from pathlib import Path


class IsaacWebRTCClient:
    def __init__(self, host: str = "localhost", port: int = 8211):
        self.host = host
        self.port = port
        self.websocket = None
        self.running = False
        
    async def connect(self):
        """Connect to Isaac Sim WebRTC service"""
        uri = f"ws://{self.host}:{self.port}/streaming/websocket"
        
        try:
            print(f"Connecting to Isaac Sim WebRTC at {uri}")
            self.websocket = await websockets.connect(uri)
            self.running = True
            print("✅ Connected to Isaac Sim WebRTC!")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    async def send_webrtc_command(self, command: str, params: dict = None):
        """Send WebRTC-specific commands to Isaac Sim"""
        if not self.websocket:
            return False
            
        message = {
            "type": command,
            "params": params or {}
        }
        
        try:
            await self.websocket.send(json.dumps(message))
            print(f"📤 Sent command: {command}")
            return True
        except Exception as e:
            print(f"❌ Failed to send command {command}: {e}")
            return False
    
    async def handle_webrtc_message(self, data):
        """Handle WebRTC-specific messages from Isaac Sim"""
        msg_type = data.get("type", "unknown")
        
        if msg_type == "webrtc_offer":
            print("📺 Received WebRTC offer from Isaac Sim")
            await self.handle_webrtc_offer(data)
        elif msg_type == "ice_candidate":
            print("🧊 Received ICE candidate")
        elif msg_type == "stream_ready":
            print("🚀 Stream is ready!")
        elif msg_type == "error":
            print(f"❌ Server error: {data.get('message', 'Unknown error')}")
        elif msg_type == "stats":
            self.print_stream_stats(data.get("data", {}))
        else:
            print(f"📨 Received: {msg_type} - {data}")
    
    async def handle_webrtc_offer(self, offer_data):
        """Handle WebRTC offer (simplified - in real use, you'd set up WebRTC peer connection)"""
        print("📺 WebRTC Offer received:")
        print(f"   SDP Type: {offer_data.get('sdp', {}).get('type', 'N/A')}")
        print("   For full WebRTC implementation, use a WebRTC library like aiortc")
        
        # Send back a simplified answer (in real implementation, create proper SDP answer)
        await self.send_webrtc_command("webrtc_answer", {
            "sdp": {
                "type": "answer",
                "sdp": "placeholder_answer"
            }
        })
    
    def print_stream_stats(self, stats):
        """Print streaming statistics"""
        if not stats:
            return
            
        print("\n📊 Stream Statistics:")
        print(f"   FPS: {stats.get('fps', 'N/A')}")
        print(f"   Bitrate: {stats.get('bitrate', 'N/A')} bps")
        print(f"   Resolution: {stats.get('width', 'N/A')}x{stats.get('height', 'N/A')}")
        print(f"   Encoder: {stats.get('encoder', 'N/A')}")
        print(f"   Clients: {stats.get('client_count', 'N/A')}")
    
    async def configure_stream(self, fps: int = 30, bitrate: int = 5000000, width: int = 1280, height: int = 720):
        """Configure stream parameters"""
        print(f"🔧 Configuring stream: {fps}fps, {bitrate}bps, {width}x{height}")
        
        await self.send_webrtc_command("configure_stream", {
            "fps": fps,
            "bitrate": bitrate,
            "width": width,
            "height": height
        })
    
    async def request_keyframe(self):
        """Request a keyframe"""
        await self.send_webrtc_command("request_keyframe")
    
    async def get_stream_info(self):
        """Get current stream information"""
        await self.send_webrtc_command("get_stream_info")
    
    async def listen_for_messages(self):
        """Main message listening loop"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self.handle_webrtc_message(data)
                except json.JSONDecodeError:
                    print(f"📝 Non-JSON message: {message[:100]}...")
                    
        except websockets.exceptions.ConnectionClosed:
            print("🔌 Connection closed by Isaac Sim")
        except Exception as e:
            print(f"❌ Error in message loop: {e}")
    
    async def interactive_mode(self):
        """Interactive command mode"""
        print("\n🎮 Interactive Mode - Available commands:")
        print("   'info' - Get stream info")
        print("   'config <fps> <bitrate>' - Configure stream")
        print("   'keyframe' - Request keyframe")
        print("   'quit' - Exit")
        print("   'help' - Show this help")
        
        while self.running:
            try:
                command = input("\n> ").strip().lower()
                
                if command == 'quit':
                    break
                elif command == 'info':
                    await self.get_stream_info()
                elif command == 'keyframe':
                    await self.request_keyframe()
                elif command.startswith('config'):
                    parts = command.split()
                    if len(parts) >= 3:
                        fps = int(parts[1])
                        bitrate = int(parts[2])
                        await self.configure_stream(fps=fps, bitrate=bitrate)
                    else:
                        print("Usage: config <fps> <bitrate>")
                elif command == 'help':
                    print("Available commands: info, config <fps> <bitrate>, keyframe, quit, help")
                else:
                    print(f"Unknown command: {command}")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Command error: {e}")
    
    async def run(self, interactive: bool = False):
        """Main run method"""
        if not await self.connect():
            return False
        
        try:
            # Initial setup
            await self.get_stream_info()
            await asyncio.sleep(1)
            
            if interactive:
                # Run interactive mode and message listening concurrently
                await asyncio.gather(
                    self.listen_for_messages(),
                    self.interactive_mode()
                )
            else:
                # Just listen for messages
                await self.listen_for_messages()
                
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
        finally:
            await self.disconnect()
    
    async def disconnect(self):
        """Disconnect from Isaac Sim"""
        self.running = False
        if self.websocket:
            await self.websocket.close()
            print("👋 Disconnected from Isaac Sim")


def create_webrtc_html_viewer(host: str = "localhost", port: int = 8211, output_file: str = "isaac_webrtc_viewer.html"):
    """Create HTML viewer for Isaac Sim WebRTC stream"""
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Isaac Sim WebRTC Viewer</title>
    <meta charset="utf-8">
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }}
        
        .stream-container {{
            background: rgba(255,255,255,0.1);
            padding: 30px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }}
        
        #videoElement {{
            width: 100%;
            max-width: 1280px;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            background: #000;
        }}
        
        .controls {{
            margin: 20px 0;
            text-align: center;
        }}
        
        button {{
            background: linear-gradient(45deg, #4CAF50, #45a049);
            border: none;
            color: white;
            padding: 12px 24px;
            margin: 8px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}
        
        button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }}
        
        button:disabled {{
            background: #666;
            cursor: not-allowed;
            transform: none;
        }}
        
        .status {{
            margin-top: 20px;
            padding: 15px;
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            border-left: 4px solid #4CAF50;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        
        .stat-item {{
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }}
        
        .quality-controls {{
            margin: 15px 0;
        }}
        
        select {{
            background: rgba(255,255,255,0.1);
            color: white;
            border: 1px solid rgba(255,255,255,0.3);
            padding: 8px 12px;
            border-radius: 5px;
            margin: 0 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Isaac Sim WebRTC Viewer</h1>
            <p>Real-time streaming from Isaac Sim simulation</p>
        </div>
        
        <div class="stream-container">
            <video id="videoElement" autoplay muted playsinline controls>
                Your browser doesn't support video playback.
            </video>
            
            <div class="controls">
                <button id="connectBtn" onclick="connect()">🔗 Connect</button>
                <button id="disconnectBtn" onclick="disconnect()" disabled>❌ Disconnect</button>
                <button onclick="toggleFullscreen()">🖥️ Fullscreen</button>
                <button onclick="requestKeyframe()">🔄 Keyframe</button>
            </div>
            
            <div class="quality-controls">
                <label>Quality:</label>
                <select id="qualitySelect" onchange="changeQuality()">
                    <option value="low">Low (720p, 2Mbps)</option>
                    <option value="medium" selected>Medium (1080p, 5Mbps)</option>
                    <option value="high">High (1080p, 10Mbps)</option>
                    <option value="ultra">Ultra (1080p, 20Mbps)</option>
                </select>
                
                <label>FPS:</label>
                <select id="fpsSelect" onchange="changeFPS()">
                    <option value="15">15 FPS</option>
                    <option value="30" selected>30 FPS</option>
                    <option value="60">60 FPS</option>
                </select>
            </div>
            
            <div class="status" id="status">
                📡 Status: Ready to connect
            </div>
            
            <div class="stats" id="stats" style="display: none;">
                <div class="stat-item">
                    <div class="stat-value" id="fpsValue">0</div>
                    <div>FPS</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="bitrateValue">0</div>
                    <div>Bitrate (Mbps)</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="resolutionValue">0x0</div>
                    <div>Resolution</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="latencyValue">0</div>
                    <div>Latency (ms)</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let pc = null;
        let video = document.getElementById('videoElement');
        let connectBtn = document.getElementById('connectBtn');
        let disconnectBtn = document.getElementById('disconnectBtn');
        let status = document.getElementById('status');
        let stats = document.getElementById('stats');
        
        function updateStatus(message, type = 'info') {{
            const icons = {{ info: '📡', success: '✅', error: '❌', warning: '⚠️' }};
            status.innerHTML = `${{icons[type] || '📡'}} Status: ${{message}}`;
            console.log('Status:', message);
        }}
        
        async function connect() {{
            try {{
                connectBtn.disabled = true;
                updateStatus('Connecting to Isaac Sim...', 'info');
                
                // Create WebSocket connection
                ws = new WebSocket('ws://{host}:{port}/streaming/websocket');
                
                ws.onopen = async function() {{
                    updateStatus('Connected! Setting up WebRTC...', 'success');
                    await setupWebRTC();
                }};
                
                ws.onmessage = async function(event) {{
                    try {{
                        const data = JSON.parse(event.data);
                        await handleMessage(data);
                    }} catch (e) {{
                        console.log('Non-JSON message:', event.data);
                    }}
                }};
                
                ws.onclose = function() {{
                    updateStatus('Disconnected from Isaac Sim', 'warning');
                    connectBtn.disabled = false;
                    disconnectBtn.disabled = true;
                    stats.style.display = 'none';
                }};
                
                ws.onerror = function(error) {{
                    updateStatus('Connection error: ' + error, 'error');
                    connectBtn.disabled = false;
                }};
                
            }} catch (error) {{
                updateStatus('Failed to connect: ' + error, 'error');
                connectBtn.disabled = false;
            }}
        }}
        
        async function setupWebRTC() {{
            try {{
                // Create RTCPeerConnection
                pc = new RTCPeerConnection({{
                    iceServers: [{{ urls: 'stun:stun.l.google.com:19302' }}]
                }});
                
                // Handle incoming stream
                pc.ontrack = function(event) {{
                    video.srcObject = event.streams[0];
                    updateStatus('Receiving video stream!', 'success');
                    disconnectBtn.disabled = false;
                    stats.style.display = 'grid';
                    startStatsMonitoring();
                }};
                
                // Handle ICE candidates
                pc.onicecandidate = function(event) {{
                    if (event.candidate) {{
                        sendMessage('ice_candidate', {{
                            candidate: event.candidate
                        }});
                    }}
                }};
                
                // Request offer from Isaac Sim
                sendMessage('request_offer');
                
            }} catch (error) {{
                updateStatus('WebRTC setup failed: ' + error, 'error');
            }}
        }}
        
        async function handleMessage(data) {{
            switch (data.type) {{
                case 'webrtc_offer':
                    await pc.setRemoteDescription(data.sdp);
                    const answer = await pc.createAnswer();
                    await pc.setLocalDescription(answer);
                    sendMessage('webrtc_answer', {{ sdp: answer }});
                    break;
                    
                case 'ice_candidate':
                    await pc.addIceCandidate(data.candidate);
                    break;
                    
                case 'stream_info':
                    updateStreamStats(data.data);
                    break;
                    
                case 'error':
                    updateStatus('Server error: ' + data.message, 'error');
                    break;
                    
                default:
                    console.log('Received:', data);
            }}
        }}
        
        function sendMessage(type, data = {{}}) {{
            if (ws && ws.readyState === WebSocket.OPEN) {{
                ws.send(JSON.stringify({{ type, ...data }}));
            }}
        }}
        
        function disconnect() {{
            if (pc) {{
                pc.close();
                pc = null;
            }}
            if (ws) {{
                ws.close();
                ws = null;
            }}
            video.srcObject = null;
            connectBtn.disabled = false;
            disconnectBtn.disabled = true;
            stats.style.display = 'none';
        }}
        
        function toggleFullscreen() {{
            if (!document.fullscreenElement) {{
                video.requestFullscreen();
            }} else {{
                document.exitFullscreen();
            }}
        }}
        
        function requestKeyframe() {{
            sendMessage('request_keyframe');
        }}
        
        function changeQuality() {{
            const quality = document.getElementById('qualitySelect').value;
            const settings = {{
                low: {{ width: 1280, height: 720, bitrate: 2000000 }},
                medium: {{ width: 1920, height: 1080, bitrate: 5000000 }},
                high: {{ width: 1920, height: 1080, bitrate: 10000000 }},
                ultra: {{ width: 1920, height: 1080, bitrate: 20000000 }}
            }};
            
            sendMessage('configure_stream', settings[quality]);
        }}
        
        function changeFPS() {{
            const fps = parseInt(document.getElementById('fpsSelect').value);
            sendMessage('configure_stream', {{ fps }});
        }}
        
        function updateStreamStats(stats) {{
            if (stats.fps) document.getElementById('fpsValue').textContent = stats.fps;
            if (stats.bitrate) document.getElementById('bitrateValue').textContent = (stats.bitrate / 1000000).toFixed(1);
            if (stats.width && stats.height) document.getElementById('resolutionValue').textContent = `${{stats.width}}x${{stats.height}}`;
        }}
        
        function startStatsMonitoring() {{
            setInterval(() => {{
                if (pc && pc.getStats) {{
                    pc.getStats().then(stats => {{
                        stats.forEach(report => {{
                            if (report.type === 'inbound-rtp' && report.mediaType === 'video') {{
                                // Update latency and other real-time stats
                                document.getElementById('latencyValue').textContent = report.jitter || 0;
                            }}
                        }});
                    }});
                }}
                
                // Request updated stream info
                sendMessage('get_stream_info');
            }}, 2000);
        }}
        
        // Handle page close
        window.addEventListener('beforeunload', disconnect);
    </script>
</body>
</html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Created Isaac Sim WebRTC viewer: {output_file}")
    print(f"🌐 Open this file in your browser to view the stream")
    print(f"📺 Stream URL: ws://{host}:{port}/streaming/websocket")


async def main():
    parser = argparse.ArgumentParser(description="Isaac Sim WebRTC Client")
    parser.add_argument("--host", default="localhost", help="Isaac Sim host")
    parser.add_argument("--port", type=int, default=8211, help="Isaac Sim WebRTC port")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive command mode")
    parser.add_argument("--create_viewer", action="store_true", help="Create HTML WebRTC viewer")
    parser.add_argument("--viewer_output", default="isaac_webrtc_viewer.html", help="HTML viewer output file")
    
    args = parser.parse_args()
    
    if args.create_viewer:
        create_webrtc_html_viewer(args.host, args.port, args.viewer_output)
        return
    
    client = IsaacWebRTCClient(args.host, args.port)
    
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum} - shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    await client.run(interactive=args.interactive)


if __name__ == "__main__":
    asyncio.run(main())