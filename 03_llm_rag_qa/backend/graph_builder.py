from langgraph.graph import StateGraph, START, END
from llm_wrapper import LLMResponder

from typing import List, Optional
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from tools import tool_list

from langgraph.graph.message import MessagesState
from langgraph.checkpoint.memory import InMemorySaver

class QAState(MessagesState):
    context: Optional[List[str]]


def main():
    system_prompt = """
    You are a heplful assistant. Your job is to answer the user's queries, regardless of what they ask for.
    You have a few tools at your disposal, check them if you need to find the answer.
    If you don't know the answer, just say: "Sorry I can't help you with that". That's all.
"""
    llm = LLMResponder("llama3.2:1b-instruct-fp16", system_prompt=system_prompt)
    llm.bind_tools(tool_list)

    builder = StateGraph(QAState)
    builder.add_node("generate", llm)
    builder.add_node("tools", ToolNode(tool_list))

    # Start → generate
    builder.add_edge(START, "generate")

    # generate → route dynamically
    builder.add_conditional_edges("generate", tools_condition)

    # tool → generate (loop after tool result)
    builder.add_edge("tools", "generate")
    builder.add_edge("generate", END)

    graph = builder.compile(checkpointer=InMemorySaver())

    return graph