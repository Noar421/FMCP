"""
MCP Server with FastMCP v2 - Improved Version
Requirements: pip install fastmcp
"""

import logging
from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError, ValidationError
import datetime
import math
from config import LoggingConfig, ArduinoConfig

from typing import Annotated, Literal
from pydantic import Field

from arduino_relays import ArduinoRelaysServer

# Initialize FastMCP v2 server
mcp = FastMCP("Arduino Relay Controller")
logger = logging.getLogger(LoggingConfig.logger_name)    
arduino_server = ArduinoRelaysServer()

# ============================================================================
# CORRECTION IMPORTANTE : Le paramètre ctx doit être après les paramètres normaux
# ============================================================================

@mcp.tool()
async def command_relays(
    command: Annotated[
        Literal["SET", "RESET"],
        Field(description="Command to be sent to Arduino (SET to turn ON, RESET to turn OFF)")
    ],
    relay_index: Annotated[
        int,
        Field(description="The index of relay to command (1-8)", ge=1, le=8)
    ],
    auto_reset_delay: Annotated[
        int,
        Field(description="Auto reset time in milliseconds (0 for manual reset)", ge=0, le=10000)
    ] = 0,
    ctx: Context = None  # Context doit être le dernier paramètre
) -> str:
    """
    Send commands to an Arduino device connected to 8 relays.
    
    - SET: Turn ON the relay
    - RESET: Turn OFF the relay
    - auto_reset_delay: If > 0, relay will automatically turn OFF after this delay
    
    Args:
        command: Command to be sent to Arduino (SET or RESET)
        relay_index: The index of relay to command (1-8)
        auto_reset_delay: Auto reset time in milliseconds (0 to 10000)
    
    Returns:
        Success message or error description
    """
    try:
        # Log the command
        if ctx:
            await ctx.info(f"Sending command to Arduino: {command} relay {relay_index}")
        
        logger.info(f"Command relay {relay_index}: {command} (auto_reset: {auto_reset_delay}ms)")
        
        # Build message
        msg = f"{command},{relay_index},{auto_reset_delay}"
        
        # Send to Arduino (async)
        response = await arduino_server.send_async(
            ArduinoConfig.ip,
            ArduinoConfig.port,
            msg,
            wait_response=True  # Wait for Arduino confirmation
        )
        
        # Log success
        if ctx:
            await ctx.info(f"Arduino responded: {response}")
        
        logger.info(f"Arduino relay {relay_index} command successful")
        
        # Return detailed success message
        action = "turned ON" if command == "SET" else "turned OFF"
        reset_info = f" (will auto-reset after {auto_reset_delay}ms)" if auto_reset_delay > 0 else ""
        return f"Success: Relay {relay_index} {action}{reset_info}"
        
    except ConnectionError as e:
        error_msg = f"Failed to connect to Arduino at {ArduinoConfig.ip}:{ArduinoConfig.port}"
        logger.error(error_msg)
        raise ToolError(error_msg)
    except TimeoutError as e:
        error_msg = f"Arduino connection timeout"
        logger.error(error_msg)
        raise ToolError(error_msg)
    except Exception as e:
        error_msg = f"Error commanding relay: {str(e)}"
        logger.error(error_msg)
        raise ToolError(error_msg)


# ============================================================================
# OUTIL SUPPLÉMENTAIRE : Status des relais
# ============================================================================

@mcp.tool()
async def get_relay_status(
    relay_index: Annotated[
        int,
        Field(description="The index of relay to check (1-8, or 0 for all)", ge=0, le=8)
    ] = 0,
    ctx: Context = None
) -> str:
    """
    Get the current status of one or all relays.
    
    Args:
        relay_index: The relay to check (1-8), or 0 for all relays
    
    Returns:
        Current status of the relay(s)
    """
    try:
        if ctx:
            await ctx.info(f"Querying Arduino relay status")
        
        # Send status query to Arduino (async)
        query = f"STATUS,{relay_index},0"
        response = await arduino_server.send_async(
            ArduinoConfig.ip,
            ArduinoConfig.port,
            query,
            wait_response=True  # Wait for status response
        )
        
        logger.info(f"Arduino status query successful: {response}")
        return f"Relay status: {response}"
        
    except Exception as e:
        error_msg = f"Error getting relay status: {str(e)}"
        logger.error(error_msg)
        raise ToolError(error_msg)


# ============================================================================
# RESSOURCE : Information sur le serveur Arduino
# ============================================================================

@mcp.resource("arduino://info")
def arduino_info() -> str:
    """Arduino controller information"""
    return f"""
Arduino Relay Controller Information:
- IP Address: {ArduinoConfig.ip}
- Port: {ArduinoConfig.port}
- Number of relays: 8
- Available commands: SET, RESET
- Auto-reset delay range: 0-10000 ms

Usage:
1. Use command_relays() to control individual relays
2. Use get_relay_status() to check relay states
3. SET turns a relay ON
4. RESET turns a relay OFF
5. Auto-reset automatically turns OFF relay after specified delay
"""


# ============================================================================
# PROMPT pour l'assistant
# ============================================================================

@mcp.prompt()
def assistant_prompt() -> str:
    """Prompt to configure the assistant for Arduino control"""
    return """You are a helpful assistant that controls an Arduino device with 8 relays.

Available capabilities:
- Turn relays ON (SET command)
- Turn relays OFF (RESET command)
- Set auto-reset timers (relay automatically turns OFF after delay)
- Check relay status

When the user asks to control relays:
1. Identify which relay number (1-8) to control
2. Determine if they want to turn it ON (SET) or OFF (RESET)
3. If they mention a duration, use auto_reset_delay in milliseconds
4. Use the command_relays tool with the appropriate parameters

Examples:
- "Turn on relay 3" → command_relays(command="SET", relay_index=3)
- "Turn off relay 5" → command_relays(command="RESET", relay_index=5)
- "Turn on relay 2 for 5 seconds" → command_relays(command="SET", relay_index=2, auto_reset_delay=5000)
- "Check relay 4 status" → get_relay_status(relay_index=4)

Always confirm the action taken after executing a command."""


# ============================================================================
# LOGGING HANDLERS pour debugging
# ============================================================================

#@mcp.on_startup
#@mcp.app.on_event("startup")
async def on_startup():
    """Called when the MCP server starts"""
    logger.info("="*60)
    logger.info("MCP Arduino Relay Server starting...")
    logger.info(f"Arduino IP: {ArduinoConfig.ip}")
    logger.info(f"Arduino Port: {ArduinoConfig.port}")
    logger.info("="*60)
    
    # Start Arduino UDP receiver
    arduino_server.start_receiver()
    print("✓ MCP Server initialized successfully")


#@mcp.on_shutdown
#@mcp.app.on_event("shutdown")
async def on_shutdown():
    """Called when the MCP server shuts down"""
    logger.info("MCP Arduino Relay Server shutting down...")
    # Clean up Arduino connection
    arduino_server.stop()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        filename=LoggingConfig.log_file,
        level=LoggingConfig.log_level,
        format=LoggingConfig.log_format
    )
    
    # Run server in HTTP mode
    print("Starting MCP Server on http://0.0.0.0:8000")
    print("Press Ctrl+C to stop")
    
    # 
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8000
    )