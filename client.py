"""
MCP Client to connect local Llama3 (via Transformers) to a FastMCP v2 server
Requirements: pip install fastmcp transformers torch accelerate
"""

import logging
import asyncio
import json
from typing import List, Dict, Any, Optional
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from fastmcp import Client
from fastmcp.exceptions import ToolError
from fastmcp.client.sampling import SamplingMessage, SamplingParams, RequestContext
from fastmcp.client.logging import LogMessage
from config import LoggingConfig

class MCPLlamaClient:
    def __init__(
        self, 
        model_name: str = "context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16",
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """
        Initialize MCP client with Llama3 via Transformers
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ("cuda", "cpu", or "auto")
            load_in_8bit: Load model in 8-bit quantization
            load_in_4bit: Load model in 4-bit quantization
        """
        self.model_name = model_name
        self.server_config = None
        self.tools = []
        self.conversation_history = []
        self.logger = logging.getLogger(LoggingConfig.logger_name)
        self.LOGGING_LEVEL_MAP = logging.getLevelNamesMapping()
        self.client = None

        print(f"🔄 Loading model {model_name}...")
        self.logger.info(f"Loading model: {self.model_name}")
        # Configure device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True)

        # Set pad_token to eos_token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model
        model_kwargs = {}
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = device
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            **model_kwargs
        )
        
        self.device = device
        print(f"✓ Model loaded on {device}")
        self.logger.debug(f"Model loaded on {device}")
        
    def create_client(self):
        self.logger.debug("Create MCP client")
        if not self.server_config:
            self.logger.debug("Configure server")
            self.set_server_config({
                                    "url": "http://localhost:8000/mcp"
                                })
        try:
            self.client = Client(self.server_config,
                    sampling_handler=self.sampling_handler,
                    log_handler=self.log_handler,
                    progress_handler=self.progress_handler
                    )
            self.logger.info(f"MCP client created {self.client.name}")
        except Exception as e:
            self.logger.error(f"Error [client/create_client] : {e}")
            raise e

    def set_server_config(self, server_config: Dict[str, Any]):
        """
        Set server configuration for connection
        
        Args:
            server_config: Server configuration, e.g.:
                {"command": "python", "args": ["server.py"]}
                or {"url": "http://localhost:8000/mcp"}
        """
        self.server_config = {
            "mcpServers": {
                "main": server_config
            }
        }
    
    def _build_tools_prompt(self) -> str:
        """Build a description of available tools for the prompt"""
        tools_desc = "You have access to the following tools:\n\n"
        for tool in self.tools:
            tools_desc += f"- {tool.name}: {tool.description}\n"
            if hasattr(tool, 'inputSchema') and tool.inputSchema:
                params = tool.inputSchema.get('properties', {})
                if params:
                    tools_desc += f"  Parameters: {', '.join(params.keys())}\n"
        
        tools_desc += """
\nTo use a tool, respond EXACTLY in this JSON format:
{
  "tool": "tool_name",
  "arguments": {
    "param1": "value1",
    "param2": "value2"
  }
}

If you don't need a tool, respond normally.
"""
        # tools_desc += "\n\nIMPORTANT: Only use tools when the user's request specifically requires them. "
        # tools_desc += "For general conversation, respond normally without using tools. "
        # tools_desc += "\nIf the tool fails, provide a natural response to the user without trying to do it by yourself."
        # tools_desc += "\nAfter using a tool, provide a natural response incorporating the tool's results."

        
        return tools_desc
    
    def _extract_tool_call(self, response: str) -> Optional[Dict]:
        """Extract a tool call from the LLM response"""
        import re
        self.logger.debug(f"Parsing for tool call in response: {response}")
        
        # Chercher un bloc JSON avec "tool" et "arguments"
        # json_pattern = r'\{[^{}]*"tool"[^{}]*"arguments"[^{}]*\}'
        json_pattern = r'\{[\s\S]*?"tool"[\s\S]*\}'
        matches = re.finditer(json_pattern, response, re.DOTALL)
        
        for match in matches:
            try:
                json_str = match.group()
                tool_call = json.loads(json_str)
                
                if "tool" in tool_call and "arguments" in tool_call:
                    self.logger.info(f"Tool call found: tool={tool_call['tool']}")
                    return tool_call
            except json.JSONDecodeError:
                continue
        
        self.logger.debug("No valid tool call found in response")
        return None
    
    async def _execute_tool_call(self, client: Client, tool_name: str, arguments: Dict) -> str:
        """Execute a tool call via FastMCP v2 server"""
        try:
            # With FastMCP v2, use call_tool with server prefix
            result = await client.call_tool(f"{tool_name}", arguments)
            self.logger.info(f"Execute tool : [{tool_name}] with {arguments}")
            
            # Extract content from result
            if hasattr(result, 'content') and result.content:
                if isinstance(result.content, list) and len(result.content) > 0:
                    return result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
                return str(result.content)
            return str(result)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
    
    def _generate_response(self, messages: List[Dict[str, str]], max_new_tokens: int = 512) -> str:
        """Generate a response with Llama3 model"""
        # Build prompt in Llama3 Instruct format
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                prompt += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "user":
                prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "assistant":
                prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
        
        # Add beginning of assistant response
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only new tokens
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    async def list_tools(self):
        if not self.server_config:
            raise ValueError("Server configuration not set. Call set_server_config() first.")
        
        if not self.client:
            self.create_client()

        async with self.client as client:
            # Get available tools list on first connection
            if not self.tools:
                self.logger.info("Fetching tools from server...")
                try:
                    self.tools = await client.list_tools()
                    print(f"✓ Connected to MCP server. {len(self.tools)} tools available:")
                    for tool in self.tools:
                        # print(f"  - {tool.name}: {tool.description}")
                        print(f"  - {tool.name}")
                except Exception as e:
                    self.logger.error(f"Failed to list tools: {e}")
                    print(f"⚠️  Warning: Could not fetch tools from server")
                    self.tools = []

    async def chat(self, user_message: str, max_iterations: int = 5):
        """Handle a conversation with tool calls"""
        if not self.server_config:
            raise ValueError("Server configuration not set. Call set_server_config() first.")
        
        if not self.client:
            self.create_client()

        self.logger.debug("[client/chat] : We have a client, let's communicate")

        async with self.client as client:
            # Get available tools list on first connection
            if not self.tools:
                self.logger.info("Fetching tools from server...")
                try:
                    self.tools = await client.list_tools()
                    print(f"✓ Connected to MCP server. {len(self.tools)} tools available:")
                    # for tool in self.tools:
                        # print(f"  - {tool.name}: {tool.description}")
                        # print(f"  - {tool.name}")
                except Exception as e:
                    self.logger.error(f"Failed to list tools: {e}")
                    print(f"⚠️  Warning: Could not fetch tools from server")
                    self.tools = []
            
            # Prepare messages
            messages = [
                {
                    "role": "system",
                    "content": f"You are a helpful assistant. {self._build_tools_prompt()}"
                }
            ]
            
            # Add history (keep last 10 exchanges to avoid context overflow)
            messages.extend(self.conversation_history[-20:])
            
            # Add new message
            messages.append({
                "role": "user",
                "content": user_message
            })
            
            # Track if we're waiting for final response after tool execution
            waiting_for_final_response = False

            for iteration in range(max_iterations):
                print(f"\n--- Iteration {iteration + 1} ---")
                self.logger.debug(f"[client/chat] : Iteration {iteration + 1}")
                
                try:
                    # Generate response
                    response = self._generate_response(messages)
                    self.logger.debug(f"[client/chat] : _generate_response = {response}")

                    # Don't look for tool calls if we're processing a tool result
                    if waiting_for_final_response:
                        self.logger.info("Processing tool result - expecting final response")
                        tool_call = None
                    else:
                        # Check for tool call
                        tool_call = self._extract_tool_call(response)                    
                    
                    if tool_call:
                        tool_name = tool_call["tool"]
                        arguments = tool_call["arguments"]
                        
                        print(f"🔧 Calling tool: {tool_name}")
                        print(f"   Arguments: {json.dumps(arguments, indent=2)}")
                        self.logger.debug(f"[client/chat] : Found tool call {tool_name}")
                        
                        # Execute tool
                        tool_result = await self._execute_tool_call(client, tool_name, arguments)
                        print(f"   Result: {tool_result[:500]}...")
                        
                        # Add to messages for next iteration
                        messages.append({
                            "role": "assistant",
                            "content": response
                        })
                        messages.append({
                            "role": "user",
                            "content": (
                                f"[TOOL EXECUTION RESULT]\n"
                                f"Tool: {tool_name}\n"
                                f"Status: Success\n"
                                f"Output: {tool_result}\n\n"
                                f"Now provide a clear, natural language response to the user "
                                f"explaining what was done. Do NOT call another tool."
                            )
                        })
                    
                        # Mark that we're waiting for the final response
                        waiting_for_final_response = True
                    
                    else:
                        # No tool call, this is the final response
                        print(f"\n🤖 Assistant: {response}")
                        self.logger.debug(f"[client/chat] : No more tool call")
                        
                        # Save to history
                        self.conversation_history.append({
                            "role": "user",
                            "content": user_message
                        })
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": response
                        })
                        
                        return response
                        
                except Exception as e:
                    self.logger.error(f"Error in iteration {iteration + 1}: {e}")
                    print(f"⚠️  Error: {e}")
                    # Continue to next iteration or break if critical
                    if iteration == max_iterations - 1:
                        return f"Error occurred: {str(e)}"
            
            return "Maximum iterations reached without final answer"

    async def sampling_handler(self,
        messages: list[SamplingMessage],
        params: SamplingParams,
        context: RequestContext
    ) -> str:
        """Handle sampling requests from the MCP server"""
        self.logger.info("Received sampling request from server")
        
        # Convert MCP messages to our format
        converted_messages = []
        for message in messages:
            content = message.content.text if hasattr(message.content, 'text') else str(message.content)
            converted_messages.append({
                "role": message.role,
                "content": content
            })
        
        # Generate response
        response = self._generate_response(converted_messages, max_new_tokens=params.maxTokens or 512)
        self.logger.info(f"Generated response for server (length: {len(response)})")
        
        return response

    async def log_handler(self,
                        message: LogMessage):
        """
        Handles incoming logs from the MCP server and forwards them
        to the standard Python logging system.
        """
        msg = message.data.get('msg')
        extra = message.data.get('extra')

        # Convert the MCP log level to a Python log level
        level = self.LOGGING_LEVEL_MAP.get(message.level.upper(), logging.INFO)

        # Log the message using the standard logging library
        self.logger.log(level, msg, extra=extra)

    async def progress_handler(
        self,
        progress: float, 
        total: float | None, 
        message: str | None
    ) -> None:
        if total is not None:
            percentage = (progress / total) * 100
            self.logger.info(f"Progress: {percentage:.1f}% - {message or ''}")
        else:
            self.logger.info(f"Progress: {progress} - {message or ''}")

async def main():
    #====================================================
    # Initialize main logger
    #====================================================
    logging.basicConfig(
        filename=LoggingConfig.log_file,
        level=LoggingConfig.log_level,
        format=LoggingConfig.log_format
    )
    logger = logging.getLogger(LoggingConfig.logger_name)    

    #====================================================
    # Initialize client
    #====================================================
    logger.info("Starting LLM Client")
    client = MCPLlamaClient(
        model_name="context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16",
        # model_name="croissantllm/CroissantLLMChat-v0.1",
        device="auto"
    )
    
    # Set server configuration
    # Option 1: Local server via stdio
    # client.set_server_config({
    #     "command": "python",
    #     "args": ["server.py"]
    # })
    
    #====================================================
    # Configure server connection
    #====================================================
    # Option 2: HTTP/SSE server
    client.set_server_config({
        "url": "http://localhost:8000/mcp"
    })
    
    #====================================================
    # Client banner
    #====================================================
    print("\n" + "="*60)
    print("MCP CLIENT + LLAMA3.2 (FastMCP v2) - Interactive mode")
    print("="*60)
    
    #====================================================
    # Client main loop
    #====================================================
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
            print(f"❌ [main] Error: {e}")

async def old_main():
    #====================================================
    # Initialize main logger
    #====================================================
    logging.basicConfig(
        filename=LoggingConfig.log_file,
        level=LoggingConfig.log_level,
        format=LoggingConfig.log_format
    )
    logger = logging.getLogger(LoggingConfig.logger_name)    

    #====================================================
    # Initialize client
    #====================================================
    logger.info("Starting LLM Client")
    client = MCPLlamaClient(
        model_name="context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16",
        device="auto"
    )
    
    # Set server configuration
    # Option 1: Local server via stdio
    # client.set_server_config({
    #     "command": "python",
    #     "args": ["server.py"]
    # })
    
    #====================================================
    # Configure server connection
    #====================================================
    # Option 2: HTTP/SSE server
    client.set_server_config({
        "url": "http://localhost:8000/mcp"
    })
    
    #====================================================
    # Client banner
    #====================================================
    print("\n" + "="*60)
    print("MCP CLIENT + LLAMA3.2 (FastMCP v2) - Interactive mode")
    print("="*60)
    
    #====================================================
    # Client main loop
    #====================================================
    while True:
        user_input = input("\n👤 You: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            break
        
        if not user_input:
            continue
        
        await client.chat(user_input)

if __name__ == "__main__":
    asyncio.run(main())