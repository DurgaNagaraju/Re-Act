<img width="440" alt="image" src="https://github.com/user-attachments/assets/673f043e-6835-483f-8aa3-1b7a03fcd7e1" />
Re-Act Agent with LangChain and LangGraph
This project demonstrates the implementation of a ReAct (Reason + Act) conversational agent using LangChain, LangGraph, OpenAI's GPT models, and various tools. The agent performs actions like querying search tools, accessing system time, and maintaining its state during interactions.

Features
ReAct Framework: Combines reasoning and acting capabilities for complex queries.

OpenAI Integration: Leverages GPT-4 for advanced natural language processing.

Tool Integration: Uses tools like TavilySearchResults to query external resources and custom tools like system time retrieval.

State Management: Maintains conversation states and tracks intermediate steps using LangGraph.

Custom Graph Execution: Implements nodes for reasoning (reason_node) and acting (act_node), utilizing LangGraph's state graphs.

Installation
Follow these steps to set up the project on your local machine.

Prerequisites
Python 3.9 or higher

An OpenAI API key

A Tavily API key (if using the TavilySearch tool)

Step 1: Clone the Repository
bash
git clone https://github.com/your_username/your_project_name.git
cd your_project_name
Step 2: Create and Activate a Virtual Environment
bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
Step 3: Install Dependencies
Install all required libraries:

bash
pip install -r requirements.txt
Example requirements.txt:

langchain
langgraph
langchain_core
langchain_openai
langchain_community
openai
python-dotenv
Step 4: Set Up API Keys
Create a .env file in the project root directory and add the following:

OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
Usage
Running the Agent
After completing the installation, run the agent script to interact with the ReAct framework:

bash
python Re-Act_Agent.py
Sample Query
The agent is preconfigured to answer queries such as:

"How many days ago was the latest SpaceX launch?"

"What is the current time in New York?"

Code Overview
1. Key Components
Tools:

TavilySearchResults: Queries external resources for search results.

get_system_time: A custom tool for retrieving the system's current date and time.

Agent:

Combines reasoning and acting using the ReAct architecture, managed by LangGraph.

State Management:

Maintains a conversation's flow and intermediate steps, enabling a seamless stateful interaction.

2. Project Structure
plaintext
├── 1_introduction/
│   ├── Re-Act_Agent.py   # Main script for the agent
│   ├── README.md         # Documentation
│   └── .env              # Environment variables
├── requirements.txt      # Project dependencies
├── venv/                 # Virtual environment
Example Code Highlights
Setting Up the Agent
python
from langchain.agents import create_react_agent
from langgraph.graph import StateGraph

# Initialize the ReAct agent
react_agent_runnable = create_react_agent(tools=tools, llm=llm, prompt=react_prompt, verbose=True)

# Define the agent state
class AgentState(TypedDict):
    input: str
    chat_history: list[tuple[str, str]]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
Using the StateGraph
python
graph = StateGraph(AgentState)
graph.add_node(REASON_NODE, reason_node)
graph.add_node(ACT_NODE, act_node)
graph.set_entry_point(REASON_NODE)

graph.add_conditional_edges(REASON_NODE, should_continue)
graph.add_edge(ACT_NODE, REASON_NODE)
Known Issues
Ensure proper API keys are set in the .env file.

Compatibility issues might occur with certain Python or library versions. Update to the latest stable versions of dependencies when possible.
