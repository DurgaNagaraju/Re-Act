import os
import json
import datetime
from apikey import apikey
from typing import Annotated, TypedDict, Union
import operator
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import tool, create_react_agent
from langchain import hub
from langchain_community.tools import TavilySearchResults
from langchain_core.agents import AgentAction, AgentFinish
from langgraph.graph import END, StateGraph

# Load environment variables
load_dotenv()

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = apikey  # Replace with your actual key

# Initialize the ChatOpenAI model
llm = ChatOpenAI(model="gpt-4")  # Replace "gpt-4" with the desired model

# Initialize the search tool
search_tool = TavilySearchResults(search_depth="basic")
react_prompt = hub.pull("hwchase17/react")

# Define a custom tool to get the system time
@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current date and time in the specified format."""
    current_time = datetime.datetime.now()
    return current_time.strftime(format)

# Combine tools into a list
tools = [search_tool, get_system_time]

# Initialize the agent
react_agent_runnable = create_react_agent(
    tools=tools,
    llm=llm,
    prompt=react_prompt,
)

# Define the AgentState class
class AgentState(TypedDict):
    input: str  # User message
    chat_history: list[tuple[str, str]]  # AI and user conversation history
    agent_outcome: Union[AgentAction, AgentFinish, None]  # Tracks agent actions or finishes
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]  # Tracks tool usage

# Define the reason node
def reason_node(state: AgentState):
    agent_outcome = react_agent_runnable.invoke(state)
    return {"agent_outcome": agent_outcome}

# Define the action node
def act_node(state: AgentState):
    agent_action = state["agent_outcome"]
    tool = next((tool for tool in tools if tool.name == agent_action.tool), None)
    if tool:
        try:
            output = tool.invoke(agent_action.tool_input)
            return {"intermediate_steps": [(agent_action, str(output))]}
        except Exception as e:
            return {"intermediate_steps": [(agent_action, f"Error: {e}")]}
    else:
        return {"intermediate_steps": [(agent_action, "Tool not found")]}

# Define constants for the graph
REASON_NODE = "reason_node"
ACT_NODE = "act_node"

# Define the graph's flow condition
def should_continue(state: AgentState) -> str:
    return END if isinstance(state["agent_outcome"], AgentFinish) else ACT_NODE

# Create a StateGraph
graph = StateGraph(AgentState)
graph.add_node(REASON_NODE, reason_node)
graph.set_entry_point(REASON_NODE)
graph.add_node(ACT_NODE, act_node)
graph.add_conditional_edges(REASON_NODE, should_continue)
graph.add_edge(ACT_NODE, REASON_NODE)

# Compile the graph
app = graph.compile()

# Execute the application
result = app.invoke(
    {
        "input": "How many days ago was the latest SpaceX launch?", 
        "agent_outcome": None,
        "intermediate_steps": []
    }
)

# Print the results
print(result)
print(result["agent_outcome"].return_values["output"], "final result")
