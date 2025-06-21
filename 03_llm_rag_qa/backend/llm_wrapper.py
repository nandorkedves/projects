from typing import Optional, List
from langchain_ollama import ChatOllama
from langgraph.graph.message import MessagesState
from langchain_core.runnables.base import Runnable
from langchain_core.messages.base import BaseMessage

class LLMResponder(Runnable):
    def __init__(self, model_name: str, system_prompt: Optional[str] = None):
        self.model_name = model_name
        self.system_prompt = system_prompt or "You are a helpful assistant."
        
        self.model = ChatOllama(model=model_name)
        
    def invoke(self, state: MessagesState, config: Optional[dict] = None):
        query = state["messages"]
        context_chunks = state.get("context", [])

        # Build context injection if needed
        if context_chunks:
            context_text = "\n\n".join(context_chunks)
            injected = [
                ("system", self.system_prompt + " Use the following context to answer the question."),
                ("user", f"Context:\n{context_text}\n\nQuestion: {query}")
            ]
            prompt = injected
        else:
            prompt = query  # just use running message history

        return {
            "messages": self.model.invoke(prompt)
        }

    def build_prompt(self, query, context=None, chat_history=None) -> List[BaseMessage]:

        messages: List[BaseMessage] = []
        messages.append(("system", self.system_prompt))
        if context:
            context_text = "\n\n".join(context)
            messages.append(("system", f"\n Use the following context to answer the query {context_text}"))
        else:
            messages.append(("user", query))

        return messages
    
    def bind_tools(self, tools):
        self.model = self.model.bind_tools(tools)