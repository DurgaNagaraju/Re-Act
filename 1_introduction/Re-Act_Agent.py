import os
import requests
from apikey import apikey
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import tool, create_react_agent
from langchain import hub
#from langchain.agents import initialize_agent, tool
import operator
from typing import Annotated, TypedDict, Union
from langchain_core.agents import AgentAction, AgentFinish
# from agent_reason_runnable import react_agent_runnable, tools
from langchain_community.tools import TavilySearchResults
from langchain_core.agents import AgentFinish, AgentAction
from langgraph.graph import END, StateGraph
#from nodes import reason_node, act_node 
# from react_state import AgentState
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
react_agent_runnable = create_react_agent(tools=[search_tool, get_system_time], llm=llm, prompt=react_prompt, verbose=True)

# Invoke the agent with a query
#agent.invoke("When was SpaceX's last launch and how many days ago was that from this instant")
#agent.invoke("What is the current time in New York?")
#agent.invoke("Who is the president of the India?")

# Define the agent state
class AgentState(TypedDict):
    # 3 state properties to achive the agent state
    input: str # HUMAN message
    chat_history: list[tuple[str, str]] # AI and HUMAN messages
    agent_outcome: Union[AgentAction, AgentFinish, None] # return the either action or finish or None
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add] #keep trcaing entire history of agent actions

# define the nodes
def reason_node(state: AgentState):
    agent_outcome = react_agent_runnable.invoke(state)
    return {"agent_outcome": agent_outcome}


tool_executor = ToolExecutor(tools)


def act_node(state: AgentState):
    agent_action = state["agent_outcome"]
    output = tool_executor.invoke(agent_action)
    return {"intermediate_steps": [(agent_action, str(output))]}

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
graph.add_node(ACT_NODE, act_node)


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