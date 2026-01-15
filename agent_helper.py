#!/usr/bin/env python3
"""
Agent Helper Module
Provides reusable functions for running ReAct agents and parsing responses.
"""

import json
import re
import uuid
from rich.console import Console

# llama-stack imports
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.react.agent import ReActAgent
from llama_stack_client.lib.agents.react.tool_parser import ReActOutput

# Suppress SSL warnings
import warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')


def extract_json_from_response(response_text):
    """
    Extract JSON from a response that might contain markdown formatting
    
    Args:
        response_text (str): Response text that might contain JSON in markdown code blocks
    
    Returns:
        dict: Parsed JSON data or None if extraction fails
    """
    try:
        # First try to parse the entire response as JSON
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code blocks
    json_patterns = [
        r'```json\s*\n(.*?)\n```',  # ```json ... ```
        r'```\s*\n(.*?)\n```',      # ``` ... ```
        r'`(.*?)`',                  # `...`
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    
    # Try to find JSON-like content in the response
    # Look for content that starts with { and ends with }
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    return None


def run_agent(system_prompt, user_prompt, tool_group, config, max_infer_iters=20):
    """
    Run a ReAct agent with custom system prompt, user prompt, and tool group
    
    Args:
        system_prompt (str): System instructions for the agent
        user_prompt (str): User prompt/query
        tool_group (str): Tool group to use (e.g., "mcp::atlassian", "mcp::openshift")
        config (dict): Configuration dictionary
        max_infer_iters (int): Maximum inference iterations (default: 20)
    
    Returns:
        dict: Response with success status, final answer, and raw response
    """
    console = Console()
    
    try:
        # Initialize client
        client = LlamaStackClient(base_url=config['base_url'])
        
        # Create ReAct agent
        tools_config = [tool_group]
        agent = ReActAgent(
            client=client,
            model=config['model'],
            tools=tools_config,
            response_format={
                "type": "json_schema",
                "json_schema": ReActOutput.model_json_schema(),
            },
            sampling_params={
                "strategy": {"type": "greedy"},
                "max_tokens": config['max_tokens'],
                "temperature": config['temperature'],
            },
            max_infer_iters=max_infer_iters
        )
        
        # Create session
        session_id = agent.create_session(session_name=f"react_agent_{uuid.uuid4()}")
        
        # Build the complete prompt
        full_prompt = f"{system_prompt}\n\nUser Request: {user_prompt}"
        
        # Create turn
        response = agent.create_turn(
            messages=[{"role": "user", "content": full_prompt}],
            session_id=session_id,
            stream=False
        )
        
        # Extract response content
        response_content = response.output_message.content
        
        # Try to extract final answer from ReAct response
        final_answer = None
        try:
            react_data = extract_json_from_response(response_content)
            if react_data and isinstance(react_data, dict):
                final_answer = react_data.get('answer', response_content)
            else:
                final_answer = response_content
        except Exception:
            final_answer = response_content
        
        return {
            "success": True,
            "final_answer": final_answer,
            "raw_response": response_content,
            "session_id": session_id,
        }
        
    except Exception as e:
        error_msg = f"ReAct Agent failed ({type(e).__name__}): {str(e)}"
        console.print(f"[red]❌ {error_msg}[/red]")
        return {
            "success": False,
            "error": error_msg,
            "final_answer": None,
            "raw_response": None,
            "session_id": None,
        }
