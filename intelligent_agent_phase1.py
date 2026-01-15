#!/usr/bin/env python3
"""
Intelligent Operations Agent (Phase-1)
A LangGraph-based workflow for automated error analysis, resolution discovery, and incident creation.

This implementation follows the LangGraph StateGraph pattern from:
https://github.com/whitew1994WW/LangGraphForBeginners/blob/main/tutorial_react.ipynb
"""

import os
from typing import Annotated, Literal, TypedDict
from datetime import datetime
from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

# Agent helper imports - choose based on environment
# For local development or when MCP servers are not directly accessible, use agent_helper
# For cluster-internal deployments with accessible MCP servers, use agent_helper_openai
import os
from dotenv import load_dotenv

# Load config to check which helper to use
load_dotenv('config.env')
USE_OPENAI_AGENT = os.getenv('USE_OPENAI_AGENT', 'false').lower() == 'true'

if USE_OPENAI_AGENT:
    try:
        from agent_helper_openai import run_agent, extract_json_from_response
        print("[INFO] Using OpenAI-compatible agent helper (agent_helper_openai)")
    except ImportError as e:
        print(f"[WARNING] Failed to import agent_helper_openai: {e}")
        print("[INFO] Falling back to llama-stack agent helper (agent_helper)")
        from agent_helper import run_agent, extract_json_from_response
else:
    from agent_helper import run_agent, extract_json_from_response
    print("[INFO] Using llama-stack agent helper (agent_helper)")


# ============================================================================
# State Definition
# ============================================================================

class WorkflowState(TypedDict):
    """State schema for the workflow graph"""
    messages: Annotated[list, add_messages]
    namespace: str
    errors_found: bool
    errors: list
    resolution_found: bool
    resolution_data: dict
    ai_resolution: dict
    confluence_page_url: str
    jira_ticket_key: str
    workflow_status: str
    error_message: str


# ============================================================================
# Configuration
# ============================================================================

def load_config():
    """Load configuration from config.env file"""
    load_dotenv('config.env')
    temperature = float(os.getenv('TEMPERATURE', '0.7'))
    if temperature <= 0:
        temperature = 0.7
    
    # Get OpenAI endpoint (preferred) or fallback to base_url
    openai_endpoint = os.getenv('OPENAI_ENDPOINT')
    base_url = os.getenv('LLAMA_STACK_URL', 'http://localhost:8321')
    
    # Ensure URL has proper protocol
    if base_url and not base_url.startswith(('http://', 'https://')):
        base_url = f'https://{base_url}'
    
    # Use OpenAI endpoint if provided, otherwise construct from base_url
    if openai_endpoint:
        # OpenAI endpoint is already configured (should include /v1/openai/v1)
        endpoint = openai_endpoint
    else:
        # Fallback: construct from base_url
        endpoint = base_url.rstrip('/')
        # Add /v1/openai/v1 for OpenAI-compatible API
        endpoint = f"{endpoint}/v1/openai/v1"
    
    # Ensure URL doesn't end with trailing slash
    endpoint = endpoint.rstrip('/')
    
    # Load MCP server URLs
    mcp_servers = {
        'openshift': os.getenv('MCP_OPENSHIFT_SERVER_URL', 'http://ocp-mcp-server:8000/sse'),
        'atlassian': os.getenv('MCP_ATLASSIAN_SERVER_URL', 'http://atlassian-mcp-server:8080/sse'),
    }
    
    # Add slack if configured
    slack_url = os.getenv('MCP_SLACK_SERVER_URL')
    if slack_url:
        mcp_servers['slack'] = slack_url
    
    return {
        'openai_endpoint': endpoint,  # For OpenAI-compatible agent
        'base_url': endpoint,  # Keep for backward compatibility
        'model': os.getenv('LLM_MODEL_ID', 'llama-4-scout-17b-16e-w4a16'),
        'api_key': os.getenv('API_KEY', 'fake'),
        'temperature': temperature,
        'max_tokens': int(os.getenv('MAX_TOKENS', '4096')),
        'timeout': int(os.getenv('LLAMA_STACK_TIMEOUT', '60')),
        'mcp_servers': mcp_servers,  # MCP server URLs for OpenAI agent
        'namespace': os.getenv('DEFAULT_NAMESPACE', 'oom-test'),
        'jira_project_key': os.getenv('JIRA_PROJECT_KEY', 'OPS'),
        'jira_issue_type': os.getenv('JIRA_ISSUE_TYPE', 'Incident'),
        'confluence_space_key': os.getenv('CONFLUENCE_SPACE_KEY', 'OPS'),
    }


# ============================================================================
# Graph Nodes
# ============================================================================

def identify_errors_node(state: WorkflowState) -> WorkflowState:
    """Node 1: Identify errors in OpenShift namespace"""
    console = Console()
    console.print(f"[bold blue]🔍 Node 1: OpenShift Error Identification[/bold blue]")
    console.print("=" * 70)
    
    config = load_config()
    namespace = state.get("namespace", config['namespace'])
    
    system_prompt = f"""You are an expert OpenShift/Kubernetes administrator. Your task is to analyze the namespace '{namespace}' for errors.

    ❗ ABSOLUTE REQUIREMENTS:
    1. YOU MUST USE THE TOOLS PROVIDED - The system will automatically execute them when you call them
    2. DO NOT write function calls as text - the tools are already available, just use them
    3. DO NOT describe what you would do - actually do it by using the tools
    4. After getting tool results, analyze them and return your findings as JSON

    🔧 HOW TO USE TOOLS:
    - The tools are already connected and ready to use
    - Simply indicate which tool you need and provide the parameters
    - The system will execute the tool and give you the results
    - Then analyze those results and return JSON

    ⚠️ CRITICAL: Your final answer must be JSON only. Do not include any tool call syntax, function names, or descriptions."""
    
    user_prompt = f"""Analyze all pods in namespace '{namespace}' for errors.

    STEPS:
    1. First, call pods_list_in_namespace with namespace='{namespace}' to get the actual list of pods
    2. For each pod you find, check its status
    3. For pods with errors (CrashLoopBackOff, Error, OOMKilled, etc.), call pods_get to get details
    4. For pods with errors, call pods_log to get error logs
    5. Call events_list to check for error events in the namespace
    6. After gathering all data, analyze and return your findings

    IMPORTANT: 
    - Use the ACTUAL pod names from the pods_list_in_namespace results, not examples
    - Return ONLY valid JSON - no markdown, no code blocks, no explanatory text
    - Start your response with {{ and end with }}

    JSON FORMAT:
    {{
        "namespace": "{namespace}",
        "errors_found": true or false,
        "error_count": number of errors found,
        "errors": [
            {{
                "pod_name": "actual pod name from the list",
                "error_type": "concise error description",
                "error_timestamp": "timestamp from events or pod status",
                "error_description": "detailed error description from logs or events",
                "pod_status": "actual pod status",
                "relevant_logs": "excerpt from logs showing the error"
            }}
        ]
    }}

    If no errors are found, return:
    {{
        "namespace": "{namespace}",
        "errors_found": false,
        "error_count": 0,
        "errors": []
    }}"""
    
    result = run_agent(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tool_group="mcp::openshift",
        config=config,
        max_infer_iters=50  # Increased to allow more tool execution iterations
    )
    
    if result['success']:
        try:
            final_answer = result.get('final_answer', '')
            tool_executed = result.get('tool_executed', False)
            tool_message_count = result.get('tool_message_count', 0)
            
            # Check if tools were actually executed
            if not tool_executed and tool_message_count == 0:
                # Tools weren't executed - this is a problem
                console.print(f"[yellow]⚠️  Warning: Tools were not executed (tool_message_count: {tool_message_count})[/yellow]")
                console.print(f"[yellow]   This may indicate the agent returned tool call info instead of executing tools[/yellow]")
            
            # Debug: Log the response if JSON parsing fails
            error_data = extract_json_from_response(final_answer)
            if error_data and isinstance(error_data, dict):
                # Validate that we got real data, not just tool call info
                if not tool_executed and ('errors_found' not in error_data or error_data.get('error_count', -1) == 0):
                    # If tools weren't executed and we got a "no errors" result, it might be invalid
                    console.print(f"[yellow]⚠️  Warning: Got 'no errors' result but tools were not executed[/yellow]")
                    console.print(f"[yellow]   Response may be unreliable. Treating as parsing failure.[/yellow]")
                    # Fall through to error handling
                    raise ValueError("Tools were not executed - response may be unreliable")
                
                console.print(f"✅ Error identification completed")
                console.print(f"   Errors found: {error_data.get('errors_found', False)}")
                console.print(f"   Error count: {error_data.get('error_count', 0)}")
                if tool_executed:
                    console.print(f"   Tools executed: {tool_message_count} tool calls")
                
                return {
                    **state,
                    "errors_found": error_data.get('errors_found', False),
                    "errors": error_data.get('errors', []),
                    "workflow_status": "errors_identified" if error_data.get('errors_found') else "no_errors",
                    "messages": state.get("messages", []) + [{"role": "assistant", "content": f"Error identification: {error_data.get('error_count', 0)} errors found"}]
                }
            else:
                # Log the actual response for debugging
                console.print("⚠️  Could not parse error identification response as JSON")
                console.print(f"[dim]Response preview (first 500 chars):[/dim]")
                console.print(f"[dim]{final_answer[:500]}[/dim]")
                
                # Try to extract partial information from the response
                # Look for common patterns that indicate errors were found
                if any(keyword in final_answer.lower() for keyword in ['error', 'failed', 'crash', 'backoff']):
                    console.print("[yellow]⚠️  Response contains error keywords but couldn't parse JSON. Attempting fallback extraction...[/yellow]")
                    # Try to create a minimal error structure
                    fallback_errors = []
                    # Look for pod names in the response
                    import re
                    pod_pattern = r'pod[_-]?name[:\s]+([a-z0-9-]+)'
                    pod_matches = re.findall(pod_pattern, final_answer, re.IGNORECASE)
                    if pod_matches:
                        for pod_name in pod_matches[:5]:  # Limit to 5 pods
                            fallback_errors.append({
                                "pod_name": pod_name,
                                "error_type": "Unknown Error",
                                "error_timestamp": "",
                                "error_description": "Error detected but details could not be parsed",
                                "pod_status": "Unknown",
                                "relevant_logs": ""
                            })
                    
                    if fallback_errors:
                        console.print(f"[yellow]⚠️  Extracted {len(fallback_errors)} potential errors from response text[/yellow]")
                        return {
                            **state,
                            "errors_found": True,
                            "errors": fallback_errors,
                            "workflow_status": "errors_identified",
                            "error_message": "Partial error extraction - JSON parsing failed",
                            "messages": state.get("messages", []) + [{"role": "assistant", "content": f"Error identification: {len(fallback_errors)} errors found (partial extraction)"}]
                        }
                
                return {
                    **state,
                    "errors_found": False,
                    "errors": [],
                    "workflow_status": "error",
                    "error_message": f"Could not parse error identification response. Response length: {len(final_answer)} chars"
                }
        except Exception as e:
            console.print(f"❌ Error parsing response: {e}")
            return {
                **state,
                "errors_found": False,
                "errors": [],
                "workflow_status": "error",
                "error_message": f"Error parsing response: {e}"
            }
    else:
        console.print(f"❌ Error identification failed: {result.get('error', 'Unknown error')}")
        return {
            **state,
            "errors_found": False,
            "errors": [],
            "workflow_status": "error",
            "error_message": result.get('error', 'Unknown error')
        }


def search_confluence_node(state: WorkflowState) -> WorkflowState:
    """Node 2: Search Confluence for resolution"""
    console = Console()
    console.print(f"[bold blue]📚 Node 2: Search Confluence for Resolution[/bold blue]")
    console.print("=" * 70)
    
    config = load_config()
    errors = state.get("errors", [])
    space_key = config['confluence_space_key']
    
    if not errors:
        return {
            **state,
            "resolution_found": False,
            "resolution_data": {},
            "workflow_status": "no_errors_to_resolve"
        }
    
    # Use the first error's type for search
    error_type = errors[0].get('error_type', 'Unknown Error')
    
    system_prompt = f"""You are an expert Confluence administrator. Your task is to search for pages and extract resolution information.

    ❗ ABSOLUTE REQUIREMENTS:
    1. YOU MUST USE THE TOOLS PROVIDED - The system will automatically execute them when you call them
    2. DO NOT write function calls as text - the tools are already available, just use them
    3. DO NOT describe what you would do - actually do it by using the tools
    4. After getting tool results, analyze them and return your findings as JSON

    🔧 HOW TO USE TOOLS:
    - The tools are already connected and ready to use
    - Call confluence_search with query and space_key parameters
    - Call confluence_get_page to retrieve full page content
    - The system will execute the tools and give you the results
    - Then analyze those results and return JSON

    ⚠️ CRITICAL: Your final answer must be JSON only. Do not include any tool call syntax, function names, or descriptions."""
    
    user_prompt = f"""Search the Confluence space '{space_key}' for a page with title containing '{error_type}' and get the page content.
    Return the resolution provided in the page without any modifications.

    IMPORTANT: Return ONLY the JSON object below. Do NOT include any markdown, code blocks, or explanatory text. Start your response with {{ and end with }}.

    {{
        "search_query": "{error_type}",
        "space_key": "{space_key}",
        "page_found": false,
        "page_title": "",
        "page_url": "",
        "resolution": "",
        "resolution_sections": []
    }}"""
    
    result = run_agent(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tool_group="mcp::atlassian",
        config=config,
        max_infer_iters=50  # Increased to allow more tool execution iterations
    )
    
    if result['success']:
        try:
            final_answer = result.get('final_answer', '')
            tool_executed = result.get('tool_executed', False)
            tool_message_count = result.get('tool_message_count', 0)
            
            # Check if tools were actually executed
            if not tool_executed and tool_message_count == 0:
                console.print(f"[yellow]⚠️  Warning: Tools were not executed (tool_message_count: {tool_message_count})[/yellow]")
                console.print(f"[yellow]   This may indicate the agent returned tool call info instead of executing tools[/yellow]")
            
            resolution_data = extract_json_from_response(final_answer)
            if resolution_data and isinstance(resolution_data, dict):
                page_found = resolution_data.get('page_found', False)
                console.print(f"✅ Resolution search completed")
                console.print(f"   Page found: {page_found}")
                if tool_executed:
                    console.print(f"   Tools executed: {tool_message_count} tool calls")
                if page_found:
                    console.print(f"   Page title: {resolution_data.get('page_title', 'Unknown')}")
                
                return {
                    **state,
                    "resolution_found": page_found,
                    "resolution_data": resolution_data,
                    "confluence_page_url": resolution_data.get('page_url', '') if page_found else '',
                    "workflow_status": "resolution_found" if page_found else "resolution_not_found",
                    "messages": state.get("messages", []) + [{"role": "assistant", "content": f"Resolution search: {'Found' if page_found else 'Not found'}"}]
                }
            else:
                console.print("⚠️  Could not parse resolution search response as JSON")
                console.print(f"[dim]Response preview (first 500 chars):[/dim]")
                console.print(f"[dim]{final_answer[:500]}[/dim]")
                
                # Return a "not found" status instead of error, so workflow can continue
                console.print("[yellow]⚠️  Treating as 'resolution not found' to continue workflow[/yellow]")
                return {
                    **state,
                    "resolution_found": False,
                    "resolution_data": {"page_found": False, "error": "Could not parse response"},
                    "workflow_status": "resolution_not_found",
                    "error_message": "Could not parse resolution search response",
                    "messages": state.get("messages", []) + [{"role": "assistant", "content": "Resolution search: Not found (parsing failed)"}]
                }
        except Exception as e:
            console.print(f"❌ Error parsing response: {e}")
            return {
                **state,
                "resolution_found": False,
                "resolution_data": {},
                "workflow_status": "error",
                "error_message": f"Error parsing response: {e}"
            }
    else:
        console.print(f"❌ Resolution search failed: {result.get('error', 'Unknown error')}")
        return {
            **state,
            "resolution_found": False,
            "resolution_data": {},
            "workflow_status": "error",
            "error_message": result.get('error', 'Unknown error')
        }


def generate_ai_resolution_node(state: WorkflowState) -> WorkflowState:
    """Node 3: Generate AI resolution when no Confluence page found"""
    console = Console()
    console.print(f"[bold blue]🤖 Node 3: Generate AI Resolution[/bold blue]")
    console.print("=" * 70)
    
    config = load_config()
    errors = state.get("errors", [])
    namespace = state.get("namespace", config['namespace'])
    
    if not errors:
        return {
            **state,
            "ai_resolution": {},
            "workflow_status": "error",
            "error_message": "No errors to generate resolution for"
        }
    
    error = errors[0]
    error_type = error.get('error_type', 'Unknown Error')
    error_description = error.get('error_description', 'No description')
    pod_name = error.get('pod_name', 'Unknown')
    
    system_prompt = "You are a Kubernetes expert. Generate a simple resolution for the error. Return ONLY valid JSON, no other text."
    user_prompt = f"""Error: {error_type} in pod {pod_name} (namespace: {namespace})
    Description: {error_description}

    Generate a resolution with:
    1. Root cause
    2. Fix steps  
    3. Verification

    Return ONLY this JSON format (no markdown, no code blocks, no other text):
    {{
        "resolution_title": "Fix for {error_type}",
        "root_cause": "brief root cause",
        "fix_steps": ["step1", "step2", "step3"],
        "verification": "how to verify fix"
    }}"""
    
    result = run_agent(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tool_group="mcp::openshift",
        config=config,
        max_infer_iters=15
    )
    
    if result['success']:
        try:
            ai_resolution_data = extract_json_from_response(result['final_answer'])
            if ai_resolution_data and isinstance(ai_resolution_data, dict):
                console.print(f"✅ AI resolution generated")
                return {
                    **state,
                    "ai_resolution": ai_resolution_data,
                    "workflow_status": "ai_resolution_generated",
                    "messages": state.get("messages", []) + [{"role": "assistant", "content": "AI resolution generated"}]
                }
            else:
                # Fallback: create simple resolution from answer
                console.print(f"✅ AI resolution generated (from answer)")
                return {
                    **state,
                    "ai_resolution": {
                        "resolution_title": f"Fix for {error_type}",
                        "root_cause": "Container not found or crashed",
                        "fix_steps": [
                            "Check pod logs for errors",
                            "Verify container image exists",
                            "Check resource limits and requests",
                            "Restart the pod if needed"
                        ],
                        "verification": "Verify pod is running and healthy"
                    },
                    "workflow_status": "ai_resolution_generated",
                    "messages": state.get("messages", []) + [{"role": "assistant", "content": "AI resolution generated"}]
                }
        except Exception as e:
            console.print(f"❌ Error parsing AI resolution: {e}")
            return {
                **state,
                "ai_resolution": {},
                "workflow_status": "error",
                "error_message": f"Error parsing AI resolution: {e}"
            }
    else:
        console.print(f"❌ AI resolution generation failed: {result.get('error', 'Unknown error')}")
        return {
            **state,
            "ai_resolution": {},
            "workflow_status": "error",
            "error_message": result.get('error', 'Unknown error')
        }


def save_to_confluence_node(state: WorkflowState) -> WorkflowState:
    """Node 4: Save AI resolution to Confluence"""
    console = Console()
    console.print(f"[bold blue]📝 Node 4: Save AI Resolution to Confluence[/bold blue]")
    console.print("=" * 70)
    
    config = load_config()
    ai_resolution = state.get("ai_resolution", {})
    space_key = config['confluence_space_key']
    
    if not ai_resolution:
        return {
            **state,
            "confluence_page_url": "",
            "workflow_status": "error",
            "error_message": "No AI resolution to save"
        }
    
    resolution_title = ai_resolution.get('resolution_title', 'AI Resolution')
    fix_steps = ai_resolution.get('fix_steps', [])
    root_cause = ai_resolution.get('root_cause', 'Unknown')
    verification = ai_resolution.get('verification', 'Unknown')
    
    # Format page content
    page_content = f"""h1. {resolution_title}

    h2. Root Cause
    {root_cause}

    h2. Resolution Steps
    {chr(10).join([f"# {step}" for step in fix_steps])}

    h2. Verification
    {verification}

    *Generated by AI on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"""
    
    system_prompt = f"""You are a Confluence expert. Your task is to create a page in Confluence.

    ❗ CRITICAL - YOU MUST ACTUALLY CALL THE TOOL:
    1. The tool name is: confluence_create_page
    2. YOU MUST CALL THIS TOOL - do not describe it, do not return it as JSON, actually CALL it
    3. When you call the tool, the system will execute it and return a result
    4. After the tool executes and returns a result, format that result as JSON

    🔧 TOOL EXECUTION PROCESS:
    Step 1: Call the tool confluence_create_page with the required parameters
    Step 2: Wait for the tool to execute and return a result
    Step 3: Take the tool's result and format it as JSON

    ⚠️ DO NOT:
    - Return tool call information as JSON
    - Describe what you would do
    - Write function signatures
    - Return {{"confluence_create_page": {{...}}}} - this is WRONG

    ✅ DO:
    - Actually call the tool confluence_create_page
    - Wait for the execution result
    - Return the execution result formatted as JSON"""
        
        user_prompt = f"""Create a Confluence page with these details:
    - Space key: {space_key}
    - Title: {resolution_title}
    - Body content: {page_content}

    IMPORTANT: Return ONLY the JSON object below. Do NOT include any markdown, code blocks, or explanatory text. Start your response with {{ and end with }}.

    {{
        "page_created": true,
        "page_title": "{resolution_title}",
        "page_url": "url from tool result"
    }}"""
    
    result = run_agent(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tool_group="mcp::atlassian",
        config=config,
        max_infer_iters=30  # Increased to allow more tool execution iterations
    )
    
    if result['success']:
        try:
            final_answer = result.get('final_answer', '')
            tool_executed = result.get('tool_executed', False)
            tool_message_count = result.get('tool_message_count', 0)
            
            # Check if tools were actually executed
            if not tool_executed and tool_message_count == 0:
                console.print(f"[yellow]⚠️  Warning: Tools were not executed (tool_message_count: {tool_message_count})[/yellow]")
                console.print(f"[yellow]   Response preview: {final_answer[:200]}...[/yellow]")
            
            confluence_data = extract_json_from_response(final_answer)
            if confluence_data and isinstance(confluence_data, dict):
                page_created = confluence_data.get('page_created', False)
                page_url = confluence_data.get('page_url', '')
                if tool_executed:
                    console.print(f"✅ Confluence page created: {page_url}")
                    console.print(f"   Tools executed: {tool_message_count} tool calls")
                else:
                    console.print(f"[yellow]⚠️  Page creation may have failed - tools not executed[/yellow]")
                
                # Update resolution_data with the created page info
                resolution_data = {
                    "page_found": True,
                    "page_title": resolution_title,
                    "page_url": page_url,
                    "resolution": f"Root Cause: {root_cause}\n\nSteps:\n" + 
                                "\n".join([f"- {step}" for step in fix_steps]) + 
                                f"\n\nVerification: {verification}",
                    "ai_generated": True
                }
                
                return {
                    **state,
                    "confluence_page_url": page_url,
                    "resolution_data": resolution_data,
                    "resolution_found": True,
                    "workflow_status": "confluence_page_created",
                    "messages": state.get("messages", []) + [{"role": "assistant", "content": f"Confluence page created: {page_url}"}]
                }
            else:
                console.print("⚠️  Could not parse Confluence creation response")
                return {
                    **state,
                    "confluence_page_url": "",
                    "workflow_status": "error",
                    "error_message": "Could not parse Confluence creation response"
                }
        except Exception as e:
            console.print(f"❌ Error parsing Confluence response: {e}")
            return {
                **state,
                "confluence_page_url": "",
                "workflow_status": "error",
                "error_message": f"Error parsing Confluence response: {e}"
            }
    else:
        console.print(f"❌ Confluence page creation failed: {result.get('error', 'Unknown error')}")
        return {
            **state,
            "confluence_page_url": "",
            "workflow_status": "error",
            "error_message": result.get('error', 'Unknown error')
        }


def create_jira_ticket_node(state: WorkflowState) -> WorkflowState:
    """Node 5: Create Jira ticket with error and resolution"""
    console = Console()
    console.print(f"[bold blue]🎫 Node 5: Create Jira Ticket[/bold blue]")
    console.print("=" * 70)
    
    config = load_config()
    errors = state.get("errors", [])
    resolution_data = state.get("resolution_data", {})
    namespace = state.get("namespace", config['namespace'])
    project_key = config['jira_project_key']
    issue_type = config['jira_issue_type']
    
    if not errors:
        return {
            **state,
            "jira_ticket_key": "",
            "workflow_status": "error",
            "error_message": "No errors to create ticket for"
        }
    
    error = errors[0]
    error_type = error.get('error_type', 'Unknown Error')
    pod_name = error.get('pod_name', 'Unknown')
    error_description = error.get('error_description', 'No description')
    resolution = resolution_data.get('resolution', 'No resolution found')
    page_title = resolution_data.get('page_title', 'Unknown Page')
    is_ai_generated = resolution_data.get('ai_generated', False)
    
    incident_title = f"{error_type} - {namespace}"
    
    ai_note = "⚠️ AI-Generated Resolution - Review before applying" if is_ai_generated else ""
    
    incident_description = f"""Pod: {pod_name}
    Namespace: {namespace}
    Error: {error_description}

    Resolution Source: {page_title}
    {resolution}

    {ai_note}"""
    
    system_prompt = f"""You are a Jira expert. Your task is to create an issue in Jira.

    ❗ CRITICAL - YOU MUST ACTUALLY CALL THE TOOL:
    1. The tool name is: jira_create_issue (NOT create_issue - must be jira_create_issue)
    2. YOU MUST CALL THIS TOOL - do not describe it, do not return it as JSON, actually CALL it
    3. When you call the tool, the system will execute it and return a result
    4. After the tool executes and returns a result, format that result as JSON

    🔧 TOOL EXECUTION PROCESS:
    Step 1: Call the tool jira_create_issue (full name, not shortened) with the required parameters
    Step 2: Wait for the tool to execute and return a result
    Step 3: Take the tool's result and format it as JSON

    ⚠️ DO NOT:
    - Use shortened names like create_issue - must use jira_create_issue
    - Return tool call information as JSON
    - Describe what you would do
    - Write function signatures

    ✅ DO:
    - Actually call the tool jira_create_issue (full name)
    - Wait for the execution result
    - Return the execution result formatted as JSON"""
        
        user_prompt = f"""Create a Jira issue with these details:
    - Project key: {project_key}
    - Issue type: {issue_type}
    - Summary: {incident_title}
    - Description: {incident_description}

    IMPORTANT: Return ONLY the JSON object below. Do NOT include any markdown, code blocks, or explanatory text. Start your response with {{ and end with }}.

    {{
        "ticket_created": true,
        "ticket_key": "issue key from tool result"
    }}"""
    
    result = run_agent(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tool_group="mcp::atlassian",
        config=config,
        max_infer_iters=30  # Increased to allow more tool execution iterations
    )
    
    if result['success']:
        try:
            final_answer = result.get('final_answer', '')
            tool_executed = result.get('tool_executed', False)
            tool_message_count = result.get('tool_message_count', 0)
            
            # Check if tools were actually executed
            if not tool_executed and tool_message_count == 0:
                console.print(f"[yellow]⚠️  Warning: Tools were not executed (tool_message_count: {tool_message_count})[/yellow]")
                console.print(f"[yellow]   Response preview: {final_answer[:200]}...[/yellow]")
            
            jira_data = extract_json_from_response(final_answer)
            if jira_data and isinstance(jira_data, dict):
                ticket_created = jira_data.get('ticket_created', False)
                ticket_key = jira_data.get('ticket_key', '')
                if tool_executed:
                    console.print(f"✅ Jira ticket created: {ticket_key}")
                    console.print(f"   Tools executed: {tool_message_count} tool calls")
                else:
                    console.print(f"[yellow]⚠️  Ticket creation may have failed - tools not executed[/yellow]")
                
                return {
                    **state,
                    "jira_ticket_key": ticket_key,
                    "workflow_status": "complete" if ticket_created else "error",
                    "error_message": "" if ticket_created else "Jira ticket creation failed",
                    "messages": state.get("messages", []) + [{"role": "assistant", "content": f"Jira ticket created: {ticket_key}"}]
                }
            else:
                console.print("⚠️  Could not parse Jira creation response")
                return {
                    **state,
                    "jira_ticket_key": "",
                    "workflow_status": "error",
                    "error_message": "Could not parse Jira creation response"
                }
        except Exception as e:
            console.print(f"❌ Error parsing Jira response: {e}")
            return {
                **state,
                "jira_ticket_key": "",
                "workflow_status": "error",
                "error_message": f"Error parsing Jira response: {e}"
            }
    else:
        console.print(f"❌ Jira ticket creation failed: {result.get('error', 'Unknown error')}")
        return {
            **state,
            "jira_ticket_key": "",
            "workflow_status": "error",
            "error_message": result.get('error', 'Unknown error')
        }


# ============================================================================
# Conditional Edge Functions
# ============================================================================

def check_errors_found(state: WorkflowState) -> Literal["__end__", "search_confluence"]:
    """Check if errors were found and route accordingly"""
    errors_found = state.get("errors_found", False)
    if errors_found:
        return "search_confluence"
    else:
        return "__end__"


def check_resolution_found(state: WorkflowState) -> Literal["generate_ai_resolution", "create_jira_ticket"]:
    """Check if resolution was found and route accordingly"""
    resolution_found = state.get("resolution_found", False)
    if resolution_found:
        return "create_jira_ticket"
    else:
        return "generate_ai_resolution"


# ============================================================================
# Graph Construction
# ============================================================================

def create_workflow_graph():
    """Create and configure the LangGraph workflow"""
    graph = StateGraph(WorkflowState)
    
    # Add nodes
    graph.add_node("identify_errors", identify_errors_node)
    graph.add_node("search_confluence", search_confluence_node)
    graph.add_node("generate_ai_resolution", generate_ai_resolution_node)
    graph.add_node("save_to_confluence", save_to_confluence_node)
    graph.add_node("create_jira_ticket", create_jira_ticket_node)
    
    # Set entry point
    graph.set_entry_point("identify_errors")
    
    # Add conditional edge: errors found?
    graph.add_conditional_edges(
        "identify_errors",
        check_errors_found,
        {
            "__end__": "__end__",
            "search_confluence": "search_confluence"
        }
    )
    
    # Add conditional edge: resolution found?
    graph.add_conditional_edges(
        "search_confluence",
        check_resolution_found,
        {
            "generate_ai_resolution": "generate_ai_resolution",
            "create_jira_ticket": "create_jira_ticket"
        }
    )
    
    # Add edge: generate AI resolution -> save to Confluence
    graph.add_edge("generate_ai_resolution", "save_to_confluence")
    
    # Add edge: save to Confluence -> create Jira ticket
    graph.add_edge("save_to_confluence", "create_jira_ticket")
    
    # Add edge: create Jira ticket -> end
    graph.add_edge("create_jira_ticket", "__end__")
    
    return graph.compile()


# ============================================================================
# Main Execution
# ============================================================================

def display_workflow_summary(state: WorkflowState):
    """Display workflow summary"""
    console = Console()
    
    table = Table(title="Workflow Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Namespace", state.get("namespace", "Unknown"))
    table.add_row("Errors Found", str(state.get("errors_found", False)))
    table.add_row("Error Count", str(len(state.get("errors", []))))
    table.add_row("Resolution Found", str(state.get("resolution_found", False)))
    table.add_row("Confluence Page URL", state.get("confluence_page_url", "N/A"))
    table.add_row("Jira Ticket Key", state.get("jira_ticket_key", "N/A"))
    table.add_row("Workflow Status", state.get("workflow_status", "Unknown"))
    
    if state.get("error_message"):
        table.add_row("Error Message", state.get("error_message", ""))
    
    console.print(table)


def main(namespace=None):
    """Main execution function"""
    console = Console()
    console.print("[bold green]🚀 INTELLIGENT OPERATIONS AGENT (PHASE-1)[/bold green]")
    console.print("=" * 70)
    
    config = load_config()
    namespace = namespace or config['namespace']
    
    # Initialize state
    initial_state: WorkflowState = {
        "messages": [{"role": "user", "content": f"Analyze errors in namespace: {namespace}"}],
        "namespace": namespace,
        "errors_found": False,
        "errors": [],
        "resolution_found": False,
        "resolution_data": {},
        "ai_resolution": {},
        "confluence_page_url": "",
        "jira_ticket_key": "",
        "workflow_status": "initialized",
        "error_message": ""
    }
    
    # Create and run workflow
    console.print(f"\n[bold yellow]Creating workflow graph...[/bold yellow]")
    app = create_workflow_graph()
    
    console.print(f"\n[bold yellow]Executing workflow for namespace: {namespace}[/bold yellow]")
    console.print("-" * 70)
    
    try:
        final_state = app.invoke(initial_state)
        
        console.print("\n" + "=" * 70)
        console.print("[bold green]✅ WORKFLOW COMPLETED![/bold green]")
        console.print("=" * 70)
        
        display_workflow_summary(final_state)
        
        return final_state
        
    except Exception as e:
        console.print(f"\n[red]❌ Workflow failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return {
            **initial_state,
            "workflow_status": "failed",
            "error_message": str(e)
        }


if __name__ == "__main__":
    import sys
    
    # Get namespace from command line argument or use default
    namespace = sys.argv[1] if len(sys.argv) > 1 else None
    
    result = main(namespace)
    
    # Exit with appropriate code
    if result.get("workflow_status") == "complete":
        sys.exit(0)
    else:
        sys.exit(1)
