from langchain_ollama import ChatOllama
from langchain.agents import create_react_agent, AgentExecutor
from src.tools import get_ticker_values, calculate_indicaters, calculate_indicators, search_ticket
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

_agent = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = create_agent()
    return _agent

def create_agent():
    llm = ChatOllama(model="qwen2.5:14b", temperature=0)
    tools = [get_ticker_values, calculate_indicaters, calculate_indicators, search_ticket]

    prompt = PromptTemplate.from_template("""You are a financial analyst assistant. Answer the following questions as best you can using the available tools.

    Chat History: {chat_history}
                                          
    You have access to the following tools:
    {tools}

    (You can use the all available tools including the ML classifier)
    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}""")

    agent = create_react_agent(llm, tools, prompt)
    memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
    )

    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        memory=memory,
        max_iterations=5,
        handle_parsing_errors=True
    )

def run(question: str):
    agent = get_agent()
    result = agent.invoke({"input": question})
    return result["output"]