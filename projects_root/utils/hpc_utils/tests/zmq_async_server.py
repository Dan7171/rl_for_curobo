import zmq
import time
import threading

class AsyncZMQServer:
    def __init__(self, port=10051):
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.REP)
        self.socket.bind(f"tcp://0.0.0.0:{port}")
        self.message_count = 0
        self.start_time = time.time()
        
        print(f"🚀 Async ZMQ REP Server listening on port {port}")
        print("Ready for pipelined requests...")
    
    def handle_request(self):
        """Handle a single request-response cycle"""
        try:
            # Receive request
            message = self.socket.recv()
            self.message_count += 1
            
            # Simulate minimal processing time
            # In real use, this would be your actual computation
            process_start = time.perf_counter()
            
            # Your actual work goes here
            # For now, just echo back
            response = b"pong"
            
            process_time = (time.perf_counter() - process_start) * 1000
            
            # Send response
            self.socket.send(response)
            
            # Log every 100 messages
            if self.message_count % 100 == 0:
                elapsed = time.time() - self.start_time
                rate = self.message_count / elapsed
                print(f"Processed {self.message_count} messages, "
                      f"Rate: {rate:.1f} msg/sec, "
                      f"Process time: {process_time:.3f}ms")
                
        except Exception as e:
            print(f"Error handling request: {e}")
    
    def run(self):
        """Main server loop"""
        print("Server running... Press Ctrl+C to stop")
        try:
            while True:
                self.handle_request()
        except KeyboardInterrupt:
            print(f"\nShutting down... Processed {self.message_count} total messages")
        finally:
            self.socket.close()
            self.ctx.term()

if __name__ == "__main__":
    server = AsyncZMQServer()
    server.run() 