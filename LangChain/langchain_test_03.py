from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain_test_02 import retriever,prompt
from base_model import llm
retriever_tool = create_retriever_tool(
    retriever,
    "CivilCode",
    ""
)
tools = [retriever_tool]
agent = create_openai_functions_agent(llm=llm,tools=tools,prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools,verbose=True)

agent_executor.invoke({"input":"建设用地使用权是什么"})