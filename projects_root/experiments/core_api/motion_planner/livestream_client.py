#!/usr/bin/env python3
"""
Livestream client for Isaac Sim WebRTC streaming
Connect to headless Isaac Sim instance to view the simulation remotely
"""

import asyncio
import websockets
import json
import argparse
import sys
import signal
from typing import Optional


class IsaacSimLivestreamClient:
    def __init__(self, host: str = "localhost", port: int = 8211):
        self.host = host
        self.port = port
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.running = False
        
    async def connect(self):
        """Connect to Isaac Sim livestream"""
        uri = f"ws://{self.host}:{self.port}/streaming/websocket"
        print(f"Connecting to Isaac Sim livestream at {uri}")
        
        try:
            self.websocket = await websockets.connect(uri)
            self.running = True
            print("Connected successfully!")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    async def send_message(self, message_type: str, data: dict = None):
        """Send control message to Isaac Sim"""
        if not self.websocket:
            return False
            
        message = {
            "type": message_type,
            "data": data or {}
        }
        
        try:
            await self.websocket.send(json.dumps(message))
            return True
        except Exception as e:
            print(f"Failed to send message: {e}")
            return False
    
    async def receive_messages(self):
        """Receive and handle messages from Isaac Sim"""
        if not self.websocket:
            return
            
        try:
            async for message in self.websocket:
                if isinstance(message, bytes):
                    # This would be video frame data
                    print(f"Received frame data: {len(message)} bytes")
                    # Here you could save frames, display them, etc.
                else:
                    # Text message
                    try:
                        data = json.loads(message)
                        await self.handle_text_message(data)
                    except json.JSONDecodeError:
                        print(f"Received non-JSON text: {message}")
                        
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed by server")
        except Exception as e:
            print(f"Error receiving messages: {e}")
    
    async def handle_text_message(self, data: dict):
        """Handle text messages from Isaac Sim"""
        message_type = data.get("type", "unknown")
        
        if message_type == "stream_info":
            print(f"Stream info: {data}")
        elif message_type == "error":
            print(f"Server error: {data.get('message', 'Unknown error')}")
        elif message_type == "status":
            print(f"Server status: {data}")
        else:
            print(f"Unknown message type '{message_type}': {data}")
    
    async def send_control_commands(self):
        """Send periodic control commands"""
        commands = [
            ("get_stream_info", {}),
            ("set_quality", {"quality": "high"}),
            ("set_framerate", {"fps": 30}),
        ]
        
        for cmd_type, cmd_data in commands:
            await self.send_message(cmd_type, cmd_data)
            await asyncio.sleep(1)
    
    async def run(self):
        """Main run loop"""
        if not await self.connect():
            return False
        
        try:
            # Send initial control commands
            await self.send_control_commands()
            
            # Start receiving messages
            await self.receive_messages()
            
        except KeyboardInterrupt:
            print("\nShutting down client...")
        finally:
            await self.disconnect()
        
        return True
    
    async def disconnect(self):
        """Disconnect from server"""
        self.running = False
        if self.websocket:
            await self.websocket.close()
            print("Disconnected from server")


class SimpleWebViewer:
    """Simple web viewer that creates an HTML page to view the stream"""
    
    @staticmethod
    def create_viewer_html(host: str = "localhost", port: int = 8211, output_file: str = "isaac_viewer.html"):
        """Create HTML file for viewing Isaac Sim livestream"""
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Isaac Sim Livestream Viewer</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background-color: #1e1e1e;
            color: white;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 20px;
        }}
        
        .stream-container {{
            text-align: center;
            background-color: #2d2d2d;
            padding: 20px;
            border-radius: 8px;
        }}
        
        .controls {{
            margin: 20px 0;
        }}
        
        button {{
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 10px 20px;
            margin: 5px;
            cursor: pointer;
            border-radius: 4px;
        }}
        
        button:hover {{
            background-color: #45a049;
        }}
        
        .status {{
            margin-top: 20px;
            padding: 10px;
            background-color: #333;
            border-radius: 4px;
        }}
        
        #streamCanvas {{
            border: 1px solid #555;
            max-width: 100%;
            height: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Isaac Sim Livestream Viewer</h1>
            <p>Connecting to: ws://{host}:{port}/streaming/websocket</p>
        </div>
        
        <div class="stream-container">
            <canvas id="streamCanvas" width="1280" height="720"></canvas>
            
            <div class="controls">
                <button onclick="connect()">Connect</button>
                <button onclick="disconnect()">Disconnect</button>
                <button onclick="toggleFullscreen()">Toggle Fullscreen</button>
            </div>
            
            <div class="status" id="status">
                Status: Not connected
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let canvas = document.getElementById('streamCanvas');
        let ctx = canvas.getContext('2d');
        let status = document.getElementById('status');
        
        function updateStatus(message) {{
            status.textContent = 'Status: ' + message;
            console.log('Status:', message);
        }}
        
        function connect() {{
            if (ws) {{
                ws.close();
            }}
            
            ws = new WebSocket('ws://{host}:{port}/streaming/websocket');
            
            ws.onopen = function(event) {{
                updateStatus('Connected to Isaac Sim');
                // Request stream info
                ws.send(JSON.stringify({{
                    type: 'get_stream_info',
                    data: {{}}
                }}));
            }};
            
            ws.onmessage = function(event) {{
                if (event.data instanceof Blob) {{
                    // Handle binary data (video frames)
                    let reader = new FileReader();
                    reader.onload = function() {{
                        let img = new Image();
                        img.onload = function() {{
                            canvas.width = img.width;
                            canvas.height = img.height;
                            ctx.drawImage(img, 0, 0);
                        }};
                        img.src = 'data:image/jpeg;base64,' + btoa(reader.result);
                    }};
                    reader.readAsBinaryString(event.data);
                }} else {{
                    // Handle text messages
                    try {{
                        let data = JSON.parse(event.data);
                        console.log('Received:', data);
                        
                        if (data.type === 'stream_info') {{
                            updateStatus('Stream active - ' + JSON.stringify(data.data));
                        }}
                    }} catch (e) {{
                        console.log('Non-JSON message:', event.data);
                    }}
                }}
            }};
            
            ws.onclose = function(event) {{
                updateStatus('Disconnected');
            }};
            
            ws.onerror = function(error) {{
                updateStatus('Error: ' + error);
            }};
        }}
        
        function disconnect() {{
            if (ws) {{
                ws.close();
                ws = null;
            }}
        }}
        
        function toggleFullscreen() {{
            if (!document.fullscreenElement) {{
                canvas.requestFullscreen();
            }} else {{
                document.exitFullscreen();
            }}
        }}
        
        // Auto-connect on page load
        window.onload = function() {{
            connect();
        }};
    </script>
</body>
</html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"Created viewer HTML file: {output_file}")
        print(f"Open this file in your browser to view the Isaac Sim livestream")


async def main():
    parser = argparse.ArgumentParser(description="Isaac Sim Livestream Client")
    parser.add_argument("--host", default="localhost", help="Isaac Sim host")
    parser.add_argument("--port", type=int, default=8211, help="Isaac Sim livestream port")
    parser.add_argument("--create_html", action="store_true", help="Create HTML viewer file")
    parser.add_argument("--html_output", default="isaac_viewer.html", help="HTML output filename")
    
    args = parser.parse_args()
    
    if args.create_html:
        SimpleWebViewer.create_viewer_html(args.host, args.port, args.html_output)
        return
    
    # Run the websocket client
    client = IsaacSimLivestreamClient(args.host, args.port)
    
    # Handle graceful shutdown
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum} - shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    await client.run()


if __name__ == "__main__":
    asyncio.run(main())