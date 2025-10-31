"""
Main launcher for MCP Arduino Relay Application
Starts either the server or client based on command line arguments
"""

import sys
import asyncio
import argparse
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import LoggingConfig, ArduinoConfig, MCPConfig, validate_all_configs
from server import on_startup, on_shutdown

def setup_argument_parser():
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description='MCP Arduino Relay Controller',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start MCP server
  python main.py server
  
  # Start interactive client
  python main.py client
  
  # Start client with custom model
  python main.py client --model meta-llama/Llama-3.2-1B-Instruct
  
  # Show configuration
  python main.py config
  
  # Test Arduino connection
  python main.py test-arduino
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['server', 'client', 'config', 'test-arduino'],
        help='Operation mode'
    )
    
    parser.add_argument(
        '--host',
        default=MCPConfig.host,
        help=f'Server host (default: {MCPConfig.host})'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=MCPConfig.port,
        help=f'Server port (default: {MCPConfig.port})'
    )
    
    parser.add_argument(
        '--model',
        default=MCPConfig.model_name,
        help=f'LLM model name (default: {MCPConfig.model_name})'
    )
    
    parser.add_argument(
        '--device',
        choices=['auto', 'cuda', 'cpu'],
        default=MCPConfig.device,
        help=f'Device to use (default: {MCPConfig.device})'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default=LoggingConfig.log_level,
        help=f'Logging level (default: {LoggingConfig.log_level})'
    )
    
    parser.add_argument(
        '--arduino-ip',
        default=ArduinoConfig.ip,
        help=f'Arduino IP address (default: {ArduinoConfig.ip})'
    )
    
    parser.add_argument(
        '--arduino-port',
        type=int,
        default=ArduinoConfig.port,
        help=f'Arduino port (default: {ArduinoConfig.port})'
    )
    
    return parser


def run_server(args):
    """Start the MCP server."""
    print("\n" + "="*60)
    print("🚀 Starting MCP Server")
    print("="*60)
    
    # Update config from args
    MCPConfig.host = args.host
    MCPConfig.port = args.port
    ArduinoConfig.ip = args.arduino_ip
    ArduinoConfig.port = args.arduino_port
    LoggingConfig.log_level = args.log_level
    
    # Setup logging
    logger = LoggingConfig.setup_logging()
    
    # Print configurations
    ArduinoConfig.print_config()
    MCPConfig.print_config()
    
    # Validate config
    if not validate_all_configs():
        print("❌ Configuration validation failed. Exiting.")
        sys.exit(1)
    
    # Import and run server
    try:
        from server import mcp
        
        logger.info(f"Starting MCP server on {args.host}:{args.port}")
        print(f"\n✓ Server will start on http://{args.host}:{args.port}")
        print(f"✓ MCP endpoint: http://{args.host}:{args.port}/mcp")
        print("\nPress Ctrl+C to stop\n")
        
        asyncio.run(on_startup())

        mcp.run(
            transport=MCPConfig.transport,
            host=args.host,
            port=args.port
        )
        
    except ImportError as e:
        print(f"❌ [main/run_server] Error importing server: {e}")
        print("Make sure 'server.py' exists and all dependencies are installed")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped by user")
        logger.info("Server stopped by user")
    except Exception as e:
        print(f"❌ [main/run_server] Server error: {e}")
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        asyncio.run(on_shutdown())



def run_client(args):
    """Start the MCP client."""
    print("\n" + "="*60)
    print("🚀 Starting MCP Client with Llama")
    print("="*60)
    
    # Update config from args
    MCPConfig.model_name = args.model
    MCPConfig.device = args.device
    MCPConfig.host = args.host
    MCPConfig.port = args.port
    LoggingConfig.log_level = args.log_level
    
    # Setup logging
    logger = LoggingConfig.setup_logging()
    
    # Print configuration
    MCPConfig.print_config()
    
    # Import and run client
    try:
        from client import MCPLlamaClient
        
        logger.info("Starting MCP client")
        
        # Create client
        client = MCPLlamaClient(
            model_name=args.model,
            device=args.device
        )
        
        # Set server configuration
        # server_url = f"http://{args.host}:{args.port}/mcp"
        server_url = f"http://127.0.0.1:{args.port}/mcp"

        client.set_server_config({"url": server_url})
        
        print(f"\n✓ Client configured to connect to {server_url}")
        print("\n" + "="*60)
        print("Interactive Mode - Type your messages")
        print("Commands: 'exit', 'quit', or 'q' to quit")
        print("="*60 + "\n")
        
        # Run interactive loop
        asyncio.run(run_interactive_client(client))
        
    except ImportError as e:
        print(f"❌ Error importing client: {e}")
        print("Make sure 'client.py' exists and all dependencies are installed")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n✓ Client stopped by user")
        logger.info("Client stopped by user")
    except Exception as e:
        print(f"❌ Client error: {e}")
        logger.error(f"Client error: {e}", exc_info=True)
        sys.exit(1)


async def run_interactive_client(client):
    """Run interactive client loop."""
    # Setup logging
    logger = LoggingConfig.setup_logging()
    logger.info(f"[main/run_interactive_client] Entering client loop")

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n✓ Goodbye!")
                break

            if user_input.lower() in ['history']:
                print(f"===== conversation_history =====")
                for h in client.conversation_history:
                    print(f"{h}")
                print(f"================================")
                continue
            
            if user_input.lower() in ['tool', 'tools']:
                print(f"============= tools ============")
                await client.list_tools()
                print(f"================================")
                continue

            if user_input.lower() in ['clear']:
                client.conversation_history = []
                print(f"Conversation history is cleared")
                continue
            
            if not user_input:
                continue
            
            # Process message
            await client.chat(user_input)
            
        except KeyboardInterrupt:
            print("\n\n✓ Exiting...")
            break
        except Exception as e:
            print(f"❌ [main/run_interactive_client] Error: {e}")


def show_config(args):
    """Show current configuration."""
    print("\n" + "="*60)
    print("📋 Current Configuration")
    print("="*60)
    
    ArduinoConfig.print_config()
    MCPConfig.print_config()
    
    print("\nLogging Configuration:")
    print("="*60)
    print(f"  Logger Name:    {LoggingConfig.logger_name}")
    print(f"  Log Level:      {LoggingConfig.log_level}")
    print(f"  Log File:       {LoggingConfig.log_file}")
    print(f"  Console Output: {LoggingConfig.log_to_console}")
    print("="*60 + "\n")
    
    # Validate
    if validate_all_configs():
        print("✓ Configuration is valid\n")
    else:
        print("❌ [main/show_config] Configuration has errors\n")


def test_arduino(args):
    """Test Arduino connection."""
    print("\n" + "="*60)
    print("🔧 Testing Arduino Connection")
    print("="*60)
    
    # Update config
    ArduinoConfig.ip = args.arduino_ip
    ArduinoConfig.port = args.arduino_port
    
    # Setup logging
    logger = LoggingConfig.setup_logging()
    
    ArduinoConfig.print_config()
    
    # Validate
    if not validate_all_configs():
        print("❌ [main/validate_all_configs] Configuration validation failed. Exiting.")
        sys.exit(1)
    
    # Test connection
    try:
        from arduino_relays import ArduinoRelaysServer
        
        print("Creating Arduino server instance...")
        arduino = ArduinoRelaysServer(
            listen_port=ArduinoConfig.server_listen_port,
            timeout=ArduinoConfig.timeout
        )
        
        print("Starting UDP receiver...")
        arduino.start_receiver()
        
        print(f"\n✓ Ready to communicate with Arduino at {ArduinoConfig.ip}:{ArduinoConfig.port}")
        print("\nAvailable test commands:")
        print("  STATUS,0,0      - Get status of all relays")
        print("  SET,1,0         - Turn ON relay 1")
        print("  RESET,1,0       - Turn OFF relay 1")
        print("  SET,2,5000      - Turn ON relay 2 for 5 seconds")
        print("  quit            - Exit test\n")
        
        while True:
            cmd = input("Command to send (or 'quit'): ").strip()
            
            if cmd.lower() == 'quit':
                break
            
            if not cmd:
                continue
            
            try:
                print(f"\n📤 Sending: {cmd}")
                response = arduino.send(
                    ArduinoConfig.ip,
                    ArduinoConfig.port,
                    cmd,
                    wait_response=True
                )
                
                if response:
                    print(f"📨 Response: {response}")
                else:
                    print("⚠️  No response received")
                    
            except Exception as e:
                print(f"❌ [main/test_arduino] Error: {e}")
        
        print("\nStopping Arduino server...")
        arduino.stop()
        print("✓ Test completed\n")
        
    except ImportError as e:
        print(f"❌ [main/test_arduino] Error importing arduino_relays: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ [main/test_arduino] Test error: {e}")
        logger.error(f"Arduino test error: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Route to appropriate function
    if args.mode == 'server':
        run_server(args)
    elif args.mode == 'client':
        run_client(args)
    elif args.mode == 'config':
        show_config(args)
    elif args.mode == 'test-arduino':
        test_arduino(args)


if __name__ == "__main__":
    main()