from json import tool
import json
import os
from matplotlib.backend_managers import ToolManagerMessageEvent, ToolMessage
import requests
from apikey import apikey
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import operator
from typing import Annotated, TypedDict, Union
from langchain.agents import create_react_agent
from langchain import hub
from langchain_community.tools import TavilySearchResults
from langchain_core.agents import AgentFinish, AgentAction
from langgraph.prebuilt.chat_agent_executor import (
    AgentState,
    Prompt,
    StateSchemaType,
    create_react_agent,
)
from langgraph.graph import END, StateGraph
import datetime

load_dotenv()

# Set OpenAI API Key (use env variable or replace with actual key)
os.environ["OPENAI_API_KEY"] = apikey

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
    formatted_time = current_time.strftime(format)
    return formatted_time

# Combine tools into a list
tools = [search_tool, get_system_time]

# Initialize the agent
react_agent_runnable = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt,
    verbose=True
)

# Define the agent state
class AgentState(TypedDict):
    input: str
    chat_history: list[tuple[str, str]]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

# Define the nodes
def reason_node(state: AgentState):
    agent_outcome = react_agent_runnable.invoke(state)
    return {"agent_outcome": agent_outcome}

def custom_tool_executor(state: AgentState):
    tools_by_name = {get_system_time.name: get_system_time}
    messages = state["messages"]
    last_message = messages[-1]
    output_messages = []
    for tool_call in last_message.tool_calls:
        try:
            tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            output_messages.append(
                ToolManagerMessageEvent(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        except Exception as e:
            output_messages.append(
                ToolMessage(
                    content="",
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                    additional_kwargs={"error": e},
                )
            )
    return {"messages": output_messages}

REASON_NODE = "reason_node"
ACT_NODE = "act_node"

def should_continue(state: AgentState) -> str:
    if isinstance(state["agent_outcome"], AgentFinish):
        return END
    else:
        return ACT_NODE

graph = StateGraph(AgentState)

graph.add_node(REASON_NODE, reason_node)
graph.set_entry_point(REASON_NODE)
graph.add_node(ACT_NODE, custom_tool_executor)

graph.add_conditional_edges(
    REASON_NODE,
    should_continue,
)

graph.add_edge(ACT_NODE, REASON_NODE)

app = graph.compile()

result = app.invoke(
    {
        "input": "How many days ago was the latest SpaceX launch?", 
        "agent_outcome": None, 
        "intermediate_steps": []
    }
)

print(result)

print(result["agent_outcome"].return_values["output"], "final result")