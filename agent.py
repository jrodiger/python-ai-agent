import operator
import uuid
import traceback
import argparse
import sys
from typing import Annotated, Sequence, TypedDict, List, Optional
from dotenv import load_dotenv
import os
from langchain_ollama import OllamaLLM
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from tools.resume_rag import resume_rag_tool
from tools.web_search import web_search_tool

# Try importing optional visualization dependencies
try:
    from IPython.display import Image
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# LangSmith Configuration
load_dotenv('.env.local')
LANGSMITH_TRACING = os.getenv('LANGSMITH_TRACING', 'true').lower() == 'true'
LANGSMITH_ENDPOINT = os.getenv('LANGSMITH_ENDPOINT', 'https://api.smith.langchain.com')
LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY', '')
LANGSMITH_PROJECT = os.getenv('LANGSMITH_PROJECT', 'python-ai-agent')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

# --- Tool Definitions ---
tools = [resume_rag_tool, web_search_tool]

# --- Agent State Definition ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# --- LLM Initialization ---
# Ensure Ollama server is running (e.g., `ollama run llama3.2`)
# Ensure the embedding model is also available (e.g., `ollama pull nomic-embed-text`)
llm = OllamaLLM(model="llama3.2") # Adjust model name if necessary

# --- Agent Logic ---

# Format tool descriptions for the prompt
tools_formatted_for_prompt = "\n".join([f"- {t.name}: {t.description}" for t in tools])

# System prompt instructing the LLM on tool usage and response format
system_prompt = f"""
You are a helpful assistant. You have access to the following tools:

{tools_formatted_for_prompt}

You must follow these rules:
1. If the question requires external data or resume information, use ONE of these tools:
   - For resume questions (experience, skills, education): start your response with 'resume_rag_tool'
   - For external information (companies, concepts, news): start your response with 'web_search_tool'
2. If you can answer from conversation history, do so directly.
3. After receiving a tool's response, you MUST provide a final answer based on that response.
   DO NOT call another tool unless absolutely necessary.

To use a tool, start your response with the tool name followed by your question.
Example: "resume_rag_tool What experience do I have with Python?"
Example: "web_search_tool What are the latest developments in AI?"

For final answers, respond with plain text (without any tool name prefix).

Remember: After using a tool, formulate a complete answer from its response. Don't chain multiple tool calls unless strictly necessary.
"""

# Create the prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Chain the prompt and LLM
agent_runnable = prompt_template | llm

def call_model(state: AgentState):
    """Invokes the LLM to decide the next action or provide a response."""
    messages = state["messages"]
    response_text = agent_runnable.invoke({"messages": messages}).strip()

    # Check if the response starts with any of our tool names
    available_tool_names = {t.name.lower() for t in tools}
    response_lower = response_text.lower()
    used_tool = None
    for tool_name in available_tool_names:
        if response_lower.startswith(tool_name):
            # Find the original tool name with correct case from tools list
            used_tool = next(t.name for t in tools if t.name.lower() == tool_name)
            break

    if used_tool:
        # It's a tool call
        tool_call_id = str(uuid.uuid4())
        # Remove the tool name from the start of the response to get the input
        # Use the length of the matched portion from the original response
        matched_prefix_length = len(response_text.split()[0])  # Get the actual tool name used in original case
        tool_input = response_text[matched_prefix_length:].strip()

        ai_msg = AIMessage(
            content="",  # Content is empty for tool calls
            tool_calls=[
                {
                    "id": tool_call_id,
                    "name": used_tool,
                    "args": {"input": tool_input}
                }
            ]
        )
    else:
        # Not a tool call, treat as final answer
        ai_msg = AIMessage(content=response_text)

    return {"messages": [ai_msg]}

# Standard ToolNode to execute the chosen tool
tool_node = ToolNode(tools)

# --- Graph Construction ---
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Define edges
workflow.set_entry_point("agent")

# Conditional edge: Route to tool execution (tools) or end based on LLM response
workflow.add_conditional_edges(
    "agent",
    # Check if the AI message contains tool calls
    lambda state: "tools" if state["messages"][-1].tool_calls else END,
    {
        "tools": "tools", # Go to tool node if tool call present
        END: END          # End the graph otherwise
    },
)

# Edge from tool execution back to the agent to process the result
workflow.add_edge("tools", "agent")

# Compile the graph
app = workflow.compile()

# --- Main Interaction Loop ---
if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='AI Agent with optional graph visualization')
    parser.add_argument('--visualize', '-v', action='store_true', 
                       help='Generate a visualization of the agent graph')
    parser.add_argument('--output', '-o', type=str, default='agent_graph.png',
                       help='Output file for graph visualization (default: agent_graph.png)')
    args = parser.parse_args()

    # Handle graph visualization if requested
    if args.visualize:
        if not VISUALIZATION_AVAILABLE:
            print("Graph visualization requires additional dependencies.")
            print("Please install them with: pip install ipython")
        else:
            try:
                print(f"Generating graph visualization to {args.output}...")
                # Remove .png extension if present in the output filename
                output_base = args.output.rsplit('.png', 1)[0]
                # Get the graph visualization as PNG data
                png_data = app.get_graph().draw_mermaid_png()
                # Save to file
                with open(f"{output_base}.png", "wb") as f:
                    f.write(png_data)
                print(f"Graph visualization saved to {output_base}.png")
                # Exit if only visualization was requested
                if not sys.stdin.isatty():  # Check if running in interactive mode
                    sys.exit(0)
            except Exception as e:
                print(f"Error generating graph visualization: {e}")
                traceback.print_exc()

    print("AI Agent Initialized. Ask me about your resume or other topics.")
    print("Using Ollama model (", llm.model, ")") # Show the model being used
    print("Available tools:", [tool.name for tool in tools])
    print("Type 'quit' or 'exit' to end the session.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        inputs = {"messages": [HumanMessage(content=user_input)]}

        try:
            # Use invoke for simpler final output display
            final_state = app.invoke(inputs)
            final_answer = final_state["messages"][-1]

            if isinstance(final_answer, AIMessage):
                # Ensure we print the content, even if the last message was technically a tool call
                # that led to this final answer after looping back.
                if final_answer.content:
                        print(f"Agent: {final_answer.content}")
                else:
                        # This might happen if the graph ends right after a tool call somehow
                        # or if the LLM returns an empty AIMessage without tool calls.
                        print("Agent: (Received an empty final response)")
            elif isinstance(final_answer, ToolMessage):
                # Should ideally not end on a ToolMessage if graph logic is correct
                # The agent should process the tool result and give a final answer
                print(f"Agent: (Ended unexpectedly after tool execution: {final_answer.content})")
            else:
                print(f"Agent: (Final state: {final_answer})")

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            traceback.print_exc() # Print full traceback for debugging
