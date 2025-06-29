from langgraph.graph import StateGraph, START, END
from llm_wrapper import LLMResponder

from typing import List, Optional
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from tools import tool_list

from langgraph.graph.message import MessagesState
from langgraph.checkpoint.memory import InMemorySaver


DEFAULT_SYSTEM_PROMPT = """
You are a helpful AI assistant.

You can:
- Answer questions directly if they are simple, conversational, or based on common knowledge.
- Use tools **only** when you truly need help — for example:
  - You don't know the answer
  - The user asks about calculations, summaries, or document-specific facts
  - If the user asks you questions about the uploaded document

DO NOT use a tool if:
- You're confident in the answer
- The question is about you, your personality, or general conversation

Be honest. If you don't know something and a tool doesn't help, say:  
"Sorry, I can't help with that."

Use tools **sparingly and only when they are the best option.**
"""

class QAState(MessagesState):
    context: Optional[List[str]]

class Graph:
    def __init__(self, model_name: str = None, thread_id: int = None, system_prompt: str = None):
        self.model_name = model_name or "qwen3:0.6b"
        self.system_prompt = system_prompt or self.get_default_system_promt()
        self.thread_id = thread_id or 42

        self.build_graph()

    def build_graph(self):
        llm = LLMResponder(self.model_name, system_prompt=self.system_prompt)
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

        self.graph = builder.compile(checkpointer=InMemorySaver())        

    def get_default_system_promt(self):
        return DEFAULT_SYSTEM_PROMPT