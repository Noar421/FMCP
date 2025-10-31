"""
Configuration module for MCP Arduino Relay Application
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class LoggingConfig:
    """Configuration for logging system"""
    
    logger_name: str = "ArduinoMCP"
    log_level: str = "DEBUG"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_file: Optional[str] = "./logs/arduino_mcp.log"
    log_format: str = (
        "%(asctime)s - %(name)-32s - %(process)-7s - "
        "%(module)-32s - %(levelname)-8s - %(message)s"
    )
    
    # Additional logging options
    log_to_console: bool = True
    log_max_bytes: int = 10 * 1024 * 1024  # 10 MB
    log_backup_count: int = 5
    
    @classmethod
    def setup_logging(cls) -> logging.Logger:
        """
        Setup and configure logging system.
        Creates log directory if needed and configures handlers.
        
        Returns:
            Configured logger instance
        """
        # Create logs directory if it doesn't exist
        if cls.log_file:
            log_path = Path(cls.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get logger
        logger = logging.getLogger(cls.logger_name)
        logger.setLevel(getattr(logging, cls.log_level.upper()))
        
        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # File handler with rotation
        if cls.log_file:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                cls.log_file,
                maxBytes=cls.log_max_bytes,
                backupCount=cls.log_backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, cls.log_level.upper()))
            file_formatter = logging.Formatter(cls.log_format)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        # Console handler (optional)
        if cls.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)  # Less verbose on console
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)-8s - %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        logger.info("="*60)
        logger.info(f"Logging system initialized: {cls.logger_name}")
        logger.info(f"Log level: {cls.log_level}")
        logger.info(f"Log file: {cls.log_file}")
        logger.info("="*60)
        
        return logger
    
    @classmethod
    def get_logger(cls) -> logging.Logger:
        """Get or create the configured logger."""
        logger = logging.getLogger(cls.logger_name)
        if not logger.handlers:
            return cls.setup_logging()
        return logger


@dataclass
class ArduinoConfig:
    """Arduino relay controller configuration"""
    
    # Network settings
    ip: str = "127.0.0.1"        # Arduino IP address
    port: int = 32000             # Arduino UDP port
    
    # Communication settings
    timeout: float = 2.0         # Socket timeout in seconds
    retry_count: int = 3         # Number of retries on failure
    retry_delay: float = 0.5     # Delay between retries in seconds
    
    # Server settings
    server_listen_ip: str = "0.0.0.0"
    server_listen_port: int = 32001  # Different from Arduino port!
    
    # Relay settings
    num_relays: int = 8
    min_relay_index: int = 1
    max_relay_index: int = 8
    min_auto_reset_delay: int = 0      # milliseconds
    max_auto_reset_delay: int = 10000  # milliseconds (10 seconds)
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate Arduino configuration.
        
        Returns:
            True if configuration is valid
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate IP address format
        ip_parts = cls.ip.split('.')
        if len(ip_parts) != 4:
            raise ValueError(f"Invalid IP address format: {cls.ip}")
        
        try:
            for part in ip_parts:
                if not 0 <= int(part) <= 255:
                    raise ValueError(f"Invalid IP address: {cls.ip}")
        except ValueError as e:
            raise ValueError(f"Invalid IP address: {cls.ip}") from e
        
        # Validate port range
        if not 1 <= cls.port <= 65535:
            raise ValueError(f"Invalid port number: {cls.port}")
        
        if not 1 <= cls.server_listen_port <= 65535:
            raise ValueError(f"Invalid server port: {cls.server_listen_port}")
        
        # Ensure Arduino and server ports are different
        if cls.port == cls.server_listen_port:
            raise ValueError(
                f"Arduino port and server port must be different! "
                f"Both are set to {cls.port}"
            )
        
        # Validate relay settings
        if cls.min_relay_index < 1:
            raise ValueError("min_relay_index must be >= 1")
        
        if cls.max_relay_index > cls.num_relays:
            raise ValueError(
                f"max_relay_index ({cls.max_relay_index}) cannot exceed "
                f"num_relays ({cls.num_relays})"
            )
        
        return True
    
    @classmethod
    def print_config(cls):
        """Print current Arduino configuration."""
        print("\n" + "="*60)
        print("Arduino Configuration:")
        print("="*60)
        print(f"  Arduino IP:        {cls.ip}")
        print(f"  Arduino Port:      {cls.port}")
        print(f"  Server Listen IP:  {cls.server_listen_ip}")
        print(f"  Server Listen Port: {cls.server_listen_port}")
        print(f"  Timeout:           {cls.timeout}s")
        print(f"  Retry Count:       {cls.retry_count}")
        print(f"  Number of Relays:  {cls.num_relays}")
        print(f"  Valid Relay Range: {cls.min_relay_index}-{cls.max_relay_index}")
        print("="*60 + "\n")


@dataclass
class MCPConfig:
    """MCP Server configuration"""
    
    # Server settings
    transport: str = "http"  # "http", "sse", or "stdio"
    host: str = "0.0.0.0"
    port: int = 8000
    
    # LLM settings
    model_name: str = "context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16"
    device: str = "auto"  # "auto", "cuda", "cpu"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Client settings
    max_tool_iterations: int = 5
    conversation_history_limit: int = 20  # Number of messages to keep
    
    @classmethod
    def print_config(cls):
        """Print current MCP configuration."""
        print("\n" + "="*60)
        print("MCP Configuration:")
        print("="*60)
        print(f"  Server Transport:  {cls.transport}")
        print(f"  Server Address:    {cls.host}:{cls.port}")
        print(f"  Model:             {cls.model_name}")
        print(f"  Device:            {cls.device}")
        print(f"  Max Tokens:        {cls.max_new_tokens}")
        print(f"  Temperature:       {cls.temperature}")
        print("="*60 + "\n")


# ============================================================================
# Configuration validation on module import
# ============================================================================

def validate_all_configs():
    """Validate all configurations."""
    try:
        ArduinoConfig.validate()
        print("✓ Configuration validated successfully")
        return True
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        return False


# ============================================================================
# Environment variable support (optional)
# ============================================================================

def load_from_env():
    """Load configuration from environment variables if present."""
    
    # Arduino config from env
    if os.getenv('ARDUINO_IP'):
        ArduinoConfig.ip = os.getenv('ARDUINO_IP')
    if os.getenv('ARDUINO_PORT'):
        ArduinoConfig.port = int(os.getenv('ARDUINO_PORT'))
    
    # Logging config from env
    if os.getenv('LOG_LEVEL'):
        LoggingConfig.log_level = os.getenv('LOG_LEVEL')
    if os.getenv('LOG_FILE'):
        LoggingConfig.log_file = os.getenv('LOG_FILE')
    
    # MCP config from env
    if os.getenv('MCP_PORT'):
        MCPConfig.port = int(os.getenv('MCP_PORT'))
    if os.getenv('MCP_MODEL'):
        MCPConfig.model_name = os.getenv('MCP_MODEL')


# Load environment variables on import
load_from_env()


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Configuration Module Test")
    print("="*60)
    
    # Print all configurations
    ArduinoConfig.print_config()
    MCPConfig.print_config()
    
    # Validate
    if validate_all_configs():
        print("\n✓ All configurations are valid")
    else:
        print("\n❌ Configuration validation failed")
    
    # Setup logging
    logger = LoggingConfig.setup_logging()
    logger.info("Configuration module test completed")
    logger.debug("This is a debug message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")