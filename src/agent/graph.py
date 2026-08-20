from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.agent.state import AgentState
from src.config import OPENAI_API_KEY
from src.tools.order_cancellation import cancel_order
from src.tools.order_status import get_order_status
from src.tools.support_docs import search_support_docs
from src.tools.support_tickets import create_support_ticket


tools = [
    get_order_status,
    search_support_docs,
    create_support_ticket,
    cancel_order,
]
model = ChatOpenAI(model="gpt-5.6-terra", api_key=OPENAI_API_KEY, reasoning_effort="none",).bind_tools(tools)


def call_model(state: AgentState) -> dict:
    response = model.invoke(state["messages"])
    return {"messages": [response]}


graph_builder = StateGraph(AgentState)
graph_builder.add_node("model", call_model)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "model")
graph_builder.add_conditional_edges("model", tools_condition)
graph_builder.add_edge("tools", "model")

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)
