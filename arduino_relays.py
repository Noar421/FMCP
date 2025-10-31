"""
Arduino Relays UDP Communication - Improved Version
Supports both synchronous and asynchronous operations
"""

import socket
import threading
import asyncio
import logging
from typing import Optional, Callable
from config import LoggingConfig

# Default Arduino configuration
ARD_IP = "127.0.0.1"
ARD_PORT = 32000


class ArduinoRelaysServer:
    def __init__(self, listen_ip="0.0.0.0", listen_port=32001, timeout=2.0):
        """
        Initialize UDP communicator for Arduino relays.
        
        Args:
            listen_ip: Local IP address to listen on
            listen_port: Local UDP port to listen on (different from Arduino port!)
            timeout: Socket timeout in seconds for send operations
        """
        self.logger = logging.getLogger(LoggingConfig.logger_name)
        self.listen_ip = listen_ip
        self.listen_port = listen_port
        self.timeout = timeout
        self.running = False
        self.thread = None
        
        # Response tracking
        self.last_response = None
        self.response_event = threading.Event()
        self.response_callback = None

        try:
            # Create UDP socket
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.settimeout(self.timeout)
            
            # Bind to local port for receiving responses
            self.sock.bind((self.listen_ip, self.listen_port))
            
            self.logger.info(
                f"Arduino UDP server initialized on {self.listen_ip}:{self.listen_port}"
            )
            print(f"✓ Arduino UDP server ready on port {self.listen_port}")
            
        except OSError as e:
            self.logger.error(f"Failed to initialize UDP socket: {e}")
            raise ConnectionError(f"Cannot bind to {self.listen_ip}:{self.listen_port}") from e

    def start_receiver(self, callback: Optional[Callable] = None):
        """
        Start UDP receiver thread.
        
        Args:
            callback: Optional function to call when data is received
                     Signature: callback(data: str, addr: tuple)
        """
        if self.running:
            self.logger.warning("Receiver already running")
            return
        
        self.running = True
        self.response_callback = callback
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        
        self.logger.info(f"UDP receiver started on {self.listen_ip}:{self.listen_port}")
        print(f"✓ UDP receiver started")

    def _receive_loop(self):
        """Internal loop to continuously receive UDP packets."""
        self.logger.info("Receiver loop started")
        
        while self.running:
            try:
                # Receive data with timeout
                data, addr = self.sock.recvfrom(4096)
                message = data.decode(errors='ignore').strip()
                
                self.logger.info(f"Received from {addr}: {message}")
                print(f"📨 [RECEIVED] From {addr}: {message}")
                
                # Store last response
                self.last_response = message
                self.response_event.set()
                
                # Call callback if provided
                if self.response_callback:
                    try:
                        self.response_callback(message, addr)
                    except Exception as e:
                        self.logger.error(f"Error in response callback: {e}")
                        
            except socket.timeout:
                # Normal timeout, continue loop
                continue
            except Exception as e:
                if self.running:  # Only log if not intentionally stopped
                    self.logger.error(f"Error in receive loop: {e}")
                break
        
        self.logger.info("Receiver loop stopped")

    def send(self, target_ip: str, target_port: int, message: str, 
             wait_response: bool = False) -> Optional[str]:
        """
        Send UDP message to Arduino.
        
        Args:
            target_ip: Target IP address (Arduino)
            target_port: Target UDP port (Arduino)
            message: Message to send
            wait_response: If True, wait for Arduino response
        
        Returns:
            Response message if wait_response=True, None otherwise
        
        Raises:
            ConnectionError: If send fails
            TimeoutError: If waiting for response times out
        """
        try:
            # Clear previous response
            if wait_response:
                self.last_response = None
                self.response_event.clear()
            
            # Send message
            self.sock.sendto(message.encode(), (target_ip, target_port))
            
            self.logger.info(f"Sent to {target_ip}:{target_port}: {message}")
            print(f"📤 [SENT] → {target_ip}:{target_port}: {message}")
            
            # Wait for response if requested
            if wait_response:
                if not self.running:
                    self.logger.warning("Receiver not running, cannot wait for response")
                    return None
                
                # Wait for response with timeout
                if self.response_event.wait(timeout=self.timeout):
                    return self.last_response
                else:
                    raise TimeoutError(
                        f"No response from Arduino within {self.timeout}s"
                    )
            
            return None
            
        except socket.error as e:
            error_msg = f"Failed to send to {target_ip}:{target_port}"
            self.logger.error(f"{error_msg}: {e}")
            raise ConnectionError(error_msg) from e

    async def send_async(self, target_ip: str, target_port: int, message: str,
                         wait_response: bool = False) -> Optional[str]:
        """
        Async version of send() for use with FastMCP.
        
        Args:
            target_ip: Target IP address (Arduino)
            target_port: Target UDP port (Arduino)
            message: Message to send
            wait_response: If True, wait for Arduino response
        
        Returns:
            Response message if wait_response=True, None otherwise
        """
        # Run synchronous send in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.send,
            target_ip,
            target_port,
            message,
            wait_response
        )

    def send_command(self, command: str, relay_index: int, 
                     auto_reset_delay: int = 0,
                     wait_response: bool = False) -> Optional[str]:
        """
        Convenience method to send relay commands.
        
        Args:
            command: "SET" or "RESET"
            relay_index: Relay number (1-8)
            auto_reset_delay: Auto-reset delay in milliseconds
            wait_response: Wait for Arduino response
        
        Returns:
            Arduino response if wait_response=True
        """
        message = f"{command},{relay_index},{auto_reset_delay}"
        return self.send(ARD_IP, ARD_PORT, message, wait_response)

    async def send_command_async(self, command: str, relay_index: int,
                                 auto_reset_delay: int = 0,
                                 wait_response: bool = False) -> Optional[str]:
        """Async version of send_command()."""
        message = f"{command},{relay_index},{auto_reset_delay}"
        return await self.send_async(ARD_IP, ARD_PORT, message, wait_response)

    def is_running(self) -> bool:
        """Check if receiver is running."""
        return self.running and self.thread is not None and self.thread.is_alive()

    def stop(self):
        """Stop receiver and close socket."""
        if not self.running:
            return
        
        self.logger.info("Stopping Arduino UDP server...")
        self.running = False
        
        # Wait for thread to finish (with timeout)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        
        # Close socket
        try:
            self.sock.close()
        except Exception as e:
            self.logger.error(f"Error closing socket: {e}")
        
        self.logger.info("Arduino UDP server stopped")
        print("✓ Arduino UDP server stopped")

    def __enter__(self):
        """Context manager entry."""
        self.start_receiver()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

    def __del__(self):
        """Destructor to ensure socket is closed."""
        if hasattr(self, 'sock'):
            try:
                self.stop()
            except:
                pass


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_callback(data: str, addr: tuple):
    """Example callback for received messages."""
    print(f"🔔 Callback triggered: {data} from {addr}")


if __name__ == "__main__":
    import logging
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("Arduino Relays UDP Communication Test")
    print("="*60 + "\n")
    
    # Create server instance
    arduino = ArduinoRelaysServer()
    arduino.start_receiver(callback=example_callback)
    
    try:
        print("\nCommands:")
        print("  SET,<relay>,<delay>   - Turn on relay")
        print("  RESET,<relay>,0       - Turn off relay")
        print("  STATUS,0,0            - Get all relay status")
        print("  quit                  - Exit")
        print()
        
        while True:
            msg = input("Command to send: ").strip()
            
            if msg.lower() == "quit":
                break
            
            if not msg:
                continue
            
            try:
                # Send with response waiting
                response = arduino.send(
                    ARD_IP, 
                    ARD_PORT, 
                    msg,
                    wait_response=True
                )
                
                if response:
                    print(f"✓ Arduino responded: {response}")
                else:
                    print("⚠️  No response from Arduino")
                    
            except TimeoutError as e:
                print(f"⏱️  Timeout: {e}")
            except ConnectionError as e:
                print(f"❌ Connection error: {e}")
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        arduino.stop()
        print("\nExiting...")